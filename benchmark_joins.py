"""
Benchmark das estratégias de junção point-in-time no Snowflake.

Compara três formas de construir o Dataset B (estado do vendedor ESTRITAMENTE
anterior a cada pedido), todas executadas no próprio Snowflake:

  1. ASOF JOIN                       (solução adotada no TCC)
  2. Subconsulta correlacionada      (abordagem SQL tradicional: MAX(ts) < data do pedido)
  3. Non-equi join + QUALIFY          (junção por faixa + ROW_NUMBER)

E, como referência da Seção 3.2 (camada de aplicação):
  4. pandas.merge_asof               (extração para a memória + junção em Python)

Cada estratégia é executada em diferentes escalas (1x = base Olist; Nx = base
replicada N vezes com vendedores distintos), com o cache de resultados
desligado. Para cada execução são coletados, do QUERY_HISTORY, o tempo total,
o tempo de execução e os bytes lidos.

Uso (na pasta do projeto, com o venv ativo):
    python benchmark_joins.py                 # escalas 1x e 10x, 5 repetições
    python benchmark_joins.py --escalas 1 10 50 --reps 5

Saídas:
    assets/benchmark_joins_execucoes.csv   (uma linha por execução)
    assets/benchmark_joins_resumo.csv      (mediana por estratégia/escala)
    assets/benchmark_joins_validacao.csv   (equivalência dos resultados)
"""
import argparse
import os
import statistics
import time

import pandas as pd

from script_ml import conectar_snowflake

FONTE = "TCC_LAKEHOUSE.GOLD.GOLD_SELLER_FEATURES"
TIMEOUT_S = 900  # consultas que passarem disso são registradas como timeout

# Créditos por hora de cada tamanho de warehouse (Snowflake Standard)
CREDITOS_HORA = {"X-SMALL": 1, "SMALL": 2, "MEDIUM": 4, "LARGE": 8, "X-LARGE": 16}

# ---------------------------------------------------------------------------
# Consultas. {F} = tabela de features na escala avaliada.
# Todas devolvem uma linha por (order_id, seller_id), com as features do estado
# imediatamente ANTERIOR ao pedido (0 se o vendedor ainda não tinha histórico).
# ---------------------------------------------------------------------------
EVENTOS = """
    eventos AS (
        SELECT order_id, seller_id, feature_timestamp AS data_do_pedido, is_delayed
        FROM {F}
    )"""

CONSULTAS = {
    "ASOF JOIN": """
WITH""" + EVENTOS + """
SELECT e.order_id, e.seller_id, e.data_do_pedido, e.is_delayed,
       COALESCE(f.seller_order_volume, 0)     AS v,
       COALESCE(f.seller_delay_count, 0)      AS d,
       COALESCE(f.seller_avg_review_score, 0) AS r
FROM eventos e
ASOF JOIN {F} f
    MATCH_CONDITION (e.data_do_pedido > f.feature_timestamp)
    ON e.seller_id = f.seller_id
""",

    "Subconsulta correlacionada": """
WITH""" + EVENTOS + """,
com_estado AS (
    SELECT e.order_id, e.seller_id, e.data_do_pedido, e.is_delayed,
           f.seller_order_volume     AS v,
           f.seller_delay_count      AS d,
           f.seller_avg_review_score AS r
    FROM eventos e
    JOIN {F} f ON f.seller_id = e.seller_id
    WHERE f.feature_timestamp = (
        SELECT MAX(f2.feature_timestamp)
        FROM {F} f2
        WHERE f2.seller_id = e.seller_id
          AND f2.feature_timestamp < e.data_do_pedido
    )
),
sem_estado AS (
    SELECT e.order_id, e.seller_id, e.data_do_pedido, e.is_delayed, 0 AS v, 0 AS d, 0 AS r
    FROM eventos e
    WHERE NOT EXISTS (
        SELECT 1 FROM {F} f2
        WHERE f2.seller_id = e.seller_id
          AND f2.feature_timestamp < e.data_do_pedido
    )
)
SELECT order_id, seller_id, data_do_pedido, is_delayed,
       COALESCE(v, 0) AS v, COALESCE(d, 0) AS d, COALESCE(r, 0) AS r
FROM (SELECT * FROM com_estado UNION ALL SELECT * FROM sem_estado)
-- empates de timestamp no mesmo vendedor: mantém uma linha por pedido
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id, seller_id ORDER BY d DESC) = 1
""",

    "Non-equi join + QUALIFY": """
WITH""" + EVENTOS + """
SELECT e.order_id, e.seller_id, e.data_do_pedido, e.is_delayed,
       COALESCE(f.seller_order_volume, 0)     AS v,
       COALESCE(f.seller_delay_count, 0)      AS d,
       COALESCE(f.seller_avg_review_score, 0) AS r
FROM eventos e
LEFT JOIN {F} f
    ON  f.seller_id = e.seller_id
    AND f.feature_timestamp < e.data_do_pedido
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY e.order_id, e.seller_id
    ORDER BY f.feature_timestamp DESC NULLS LAST) = 1
""",
}


def executar(cur, sql, params=None):
    cur.execute(sql, params)
    return cur.fetchall()


def preparar_sessao(cur):
    executar(cur, "ALTER SESSION SET USE_CACHED_RESULT = FALSE")
    executar(cur, f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {TIMEOUT_S}")
    executar(cur, "ALTER SESSION SET QUERY_TAG = 'tcc_benchmark_joins'")
    wh = executar(cur, "SELECT CURRENT_WAREHOUSE()")[0][0]
    cur.execute(f"SHOW WAREHOUSES LIKE '{wh}'")
    cols = [c[0].lower() for c in cur.description]
    tamanho = dict(zip(cols, cur.fetchone()))["size"].upper().replace("XSMALL", "X-SMALL")
    return wh, tamanho


def criar_escala(cur, n):
    """Tabela temporária com a base replicada n vezes (vendedores e pedidos distintos)."""
    nome = f"BENCH_FEATURES_{n}X"
    if n == 1:
        sql = f"CREATE OR REPLACE TEMPORARY TABLE {nome} AS SELECT * FROM {FONTE}"
    else:
        sql = f"""
        CREATE OR REPLACE TEMPORARY TABLE {nome} AS
        SELECT f.order_id  || '_' || g.rep AS order_id,
               f.is_delayed,
               f.seller_id || '_' || g.rep AS seller_id,
               f.feature_timestamp,
               f.seller_order_volume, f.seller_delay_count, f.seller_avg_review_score
        FROM {FONTE} f
        CROSS JOIN (SELECT SEQ4() AS rep FROM TABLE(GENERATOR(ROWCOUNT => {n}))) g
        """
    executar(cur, sql)
    linhas = executar(cur, f"SELECT COUNT(*) FROM {nome}")[0][0]
    return nome, linhas


def metricas_da_consulta(cur, query_id):
    # O QUERY_HISTORY pode levar alguns segundos para registrar a consulta
    for _ in range(10):
        r = executar(cur, """
            SELECT total_elapsed_time, execution_time, compilation_time, bytes_scanned
            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION(RESULT_LIMIT => 200))
            WHERE query_id = %s""", (query_id,))
        if r:
            return {"total_ms": r[0][0], "execucao_ms": r[0][1],
                    "compilacao_ms": r[0][2], "bytes_lidos": r[0][3]}
        time.sleep(2)
    return {"total_ms": None, "execucao_ms": None, "compilacao_ms": None, "bytes_lidos": None}


def medir_sql(cur, nome_estrategia, sql, tabela):
    # Agregação no final força o cálculo de todas as colunas sem trafegar linhas
    q = f"SELECT COUNT(*), SUM(v), SUM(d), SUM(r) FROM ({sql.format(F=tabela)})"
    t0 = time.perf_counter()
    try:
        cur.execute(q)
        cur.fetchall()
        status = "ok"
    except Exception as exc:  # timeout ou subconsulta não suportada
        status = "timeout" if "timeout" in str(exc).lower() else f"erro: {str(exc)[:120]}"
    parede_ms = (time.perf_counter() - t0) * 1000
    m = metricas_da_consulta(cur, cur.sfqid) if status == "ok" else {}
    return {"status": status, "parede_ms": parede_ms, **m}


def medir_pandas(ctx, tabela):
    """Extrai a tabela para a memória e aplica pandas.merge_asof (camada de aplicação)."""
    t0 = time.perf_counter()
    cur = ctx.cursor()
    cur.execute(f"SELECT * FROM {tabela}")
    df = cur.fetch_pandas_all()
    t1 = time.perf_counter()
    eventos = df[["ORDER_ID", "SELLER_ID", "FEATURE_TIMESTAMP", "IS_DELAYED"]] \
        .rename(columns={"FEATURE_TIMESTAMP": "DATA_DO_PEDIDO"}).sort_values("DATA_DO_PEDIDO")
    estados = df[["SELLER_ID", "FEATURE_TIMESTAMP", "SELLER_ORDER_VOLUME",
                  "SELLER_DELAY_COUNT", "SELLER_AVG_REVIEW_SCORE"]].sort_values("FEATURE_TIMESTAMP")
    res = pd.merge_asof(eventos, estados, left_on="DATA_DO_PEDIDO", right_on="FEATURE_TIMESTAMP",
                        by="SELLER_ID", direction="backward", allow_exact_matches=False)
    t2 = time.perf_counter()
    mem_mb = (df.memory_usage(deep=True).sum() + res.memory_usage(deep=True).sum()) / 1e6
    return {"status": "ok", "parede_ms": (t2 - t0) * 1000, "extracao_ms": (t1 - t0) * 1000,
            "juncao_ms": (t2 - t1) * 1000, "memoria_mb": mem_mb}


def validar(cur, tabela):
    """Confere se as três consultas SQL produzem o mesmo resultado que o ASOF JOIN."""
    for nome, sql in CONSULTAS.items():
        alias = "RES_" + "".join(ch for ch in nome.upper() if ch.isalnum())[:20]
        executar(cur, f"CREATE OR REPLACE TEMPORARY TABLE {alias} AS {sql.format(F=tabela)}")
    linhas = []
    ref = "RES_ASOFJOIN"
    for nome in CONSULTAS:
        alias = "RES_" + "".join(ch for ch in nome.upper() if ch.isalnum())[:20]
        n, difs = executar(cur, f"""
            SELECT (SELECT COUNT(*) FROM {alias}),
                   COUNT_IF(a.v IS DISTINCT FROM b.v OR a.d IS DISTINCT FROM b.d
                            OR a.r IS DISTINCT FROM b.r)
            FROM {ref} a FULL OUTER JOIN {alias} b
              ON a.order_id = b.order_id AND a.seller_id = b.seller_id""")[0]
        linhas.append({"estrategia": nome, "linhas": n, "linhas_diferentes_do_asof": difs})
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escalas", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sem-pandas", action="store_true")
    args = ap.parse_args()

    ctx = conectar_snowflake()
    cur = ctx.cursor()
    wh, tamanho = preparar_sessao(cur)
    print(f"Warehouse: {wh} ({tamanho}) | cache de resultados desligado | timeout {TIMEOUT_S}s\n")
    os.makedirs("assets", exist_ok=True)

    execucoes = []
    for n in args.escalas:
        tabela, linhas = criar_escala(cur, n)
        print(f"=== Escala {n}x: {linhas:,} linhas em {tabela} ===")

        if n == args.escalas[0]:
            print("Validando equivalência dos resultados...")
            val = validar(cur, tabela)
            print(val.to_string(index=False), "\n")
            val.to_csv("assets/benchmark_joins_validacao.csv", index=False)

        estrategias = list(CONSULTAS.items()) + ([] if args.sem_pandas else [("pandas.merge_asof", None)])
        for nome, sql in estrategias:
            for rep in range(args.reps):
                r = medir_pandas(ctx, tabela) if sql is None else medir_sql(cur, nome, sql, tabela)
                execucoes.append({"escala": n, "linhas": linhas, "estrategia": nome, "rep": rep + 1, **r})
                print(f"  {nome:<28} rep {rep + 1}: {r['parede_ms'] / 1000:8.2f}s  [{r['status']}]")
                if r["status"] != "ok":
                    break  # não repete consultas que estouraram o tempo
        print()

    df = pd.DataFrame(execucoes)
    df.to_csv("assets/benchmark_joins_execucoes.csv", index=False)

    # Resumo: mediana descartando a 1ª repetição (aquecimento do warehouse)
    ok = df[(df["status"] == "ok") & ((df["rep"] > 1) | (args.reps == 1))]
    resumo = ok.groupby(["escala", "linhas", "estrategia"]).agg(
        mediana_s=("parede_ms", lambda s: statistics.median(s) / 1000),
        execucao_s=("execucao_ms", lambda s: statistics.median(s.dropna()) / 1000 if s.notna().any() else None),
        bytes_lidos_mb=("bytes_lidos", lambda s: statistics.median(s.dropna()) / 1e6 if s.notna().any() else None),
        reps=("rep", "count"),
    ).reset_index()
    cph = CREDITOS_HORA.get(tamanho)
    if cph:
        resumo["creditos_estimados"] = resumo["execucao_s"] * cph / 3600
    base = resumo[resumo["estrategia"] == "ASOF JOIN"].set_index("escala")["mediana_s"]
    resumo["x_mais_lento_que_asof"] = resumo.apply(lambda r: r["mediana_s"] / base.get(r["escala"], float("nan")), axis=1)
    falhas = df[df["status"] != "ok"][["escala", "estrategia", "status"]].drop_duplicates()

    print("=== RESUMO (mediana, sem a 1ª repetição) ===")
    print(resumo.round(4).to_string(index=False))
    if not falhas.empty:
        print("\nExecuções que não terminaram:")
        print(falhas.to_string(index=False))
    resumo.to_csv("assets/benchmark_joins_resumo.csv", index=False)
    print("\nCSVs salvos em assets/")


if __name__ == "__main__":
    main()
