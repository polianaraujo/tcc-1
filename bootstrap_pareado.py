"""
Bootstrap pareado e por conglomerados (vendedor) para as comparações do capítulo de Resultados.

  1. IC 95% da ROC-AUC e da PR-AUC (average precision), por modelo e cenário, com dois esquemas
     de reamostragem do conjunto de teste:
       - por linhas (igual ao bootstrap_auc_ci do script_ml.py);
       - por vendedores (bootstrap por conglomerados: sorteia vendedores com reposição e leva
         todos os seus pedidos), que respeita a dependência entre pedidos do mesmo vendedor.
  2. IC 95% das diferenças pareadas, usando os MESMOS índices de reamostragem em todos os modelos:
       - A - B para cada modelo (efeito do vazamento);
       - Random Forest - XGBoost e Reg. Logística - XGBoost em cada cenário (comparação entre modelos).
  3. Diagnósticos auxiliares:
       - positivos com SELLER_DELAY_COUNT = 0 no Cenário A (deve ser 0 por construção, pois a
         contagem final inclui o desfecho do próprio pedido);
       - taxa de atraso por mês de compra (ajuda a avaliar a queda de 7,5% para 3,4%).

Reaproveita conexão, extração e treinamento do script_ml.py (mesma pasta); os modelos são
retreinados com as mesmas sementes e hiperparâmetros, de modo que as estimativas pontuais
coincidem com as do experimento principal.

Uso (na pasta do projeto, com o venv ativo):
    python bootstrap_pareado.py

Saídas: assets/bootstrap_pareado.csv e assets/taxa_atraso_mensal.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from script_ml import conectar_snowflake, extrair_dados, treinar_e_avaliar_completo

N_BOOT = 1000
SEED = 42
COL_DATA = 'DATA_DO_PEDIDO'
CHAVE = ['ORDER_ID', 'SELLER_ID']
TARGET = 'TARGET_HOUVE_ATRASO'
MODELOS = ['Random Forest', 'Reg. Logística', 'XGBoost']


def particoes(df, frac_treino=0.8):
    # Mesma ordenação e mesmo corte de script_ml.treinar_e_avaliar_completo
    df = df.sort_values([COL_DATA] + CHAVE).reset_index(drop=True)
    corte = int(len(df) * frac_treino)
    return df.iloc[:corte].reset_index(drop=True), df.iloc[corte:].reset_index(drop=True)


# ==========================================
# 1. AMOSTRADORES (devolvem índices de linhas do conjunto de teste)
# ==========================================
def amostrador_linhas(n):
    return lambda rng: rng.randint(0, n, n)


def amostrador_vendedores(vendedores):
    codigos, _ = pd.factorize(pd.Series(vendedores))
    ordem = np.argsort(codigos, kind='stable')
    cortes = np.cumsum(np.bincount(codigos))[:-1]
    membros = np.split(ordem, cortes)          # membros[g] = linhas do vendedor g
    g = len(membros)
    return lambda rng: np.concatenate([membros[i] for i in rng.randint(0, g, g)])


# ==========================================
# 2. BOOTSTRAP PAREADO
# ==========================================
def bootstrap(y, probs, amostrar, n_boot=N_BOOT, seed=SEED):
    """probs: {(cenario, modelo): probabilidades}. Todos os modelos usam os mesmos índices."""
    rng = np.random.RandomState(seed)
    auc = {k: [] for k in probs}
    ap = {k: [] for k in probs}
    for _ in range(n_boot):
        idx = amostrar(rng)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k, p in probs.items():
            auc[k].append(roc_auc_score(yb, p[idx]))
            ap[k].append(average_precision_score(yb, p[idx]))
    return ({k: np.array(v) for k, v in auc.items()},
            {k: np.array(v) for k, v in ap.items()})


def _ic(x, nivel=0.95):
    a = (1 - nivel) / 2 * 100
    return np.percentile(x, a), np.percentile(x, 100 - a)


def _linha(metrica, amostragem, comparacao, estimativa, reps, p=True):
    lo, hi = _ic(reps)
    p_bil = min(1.0, 2 * min(np.mean(reps <= 0), np.mean(reps >= 0))) if p else np.nan
    return dict(metrica=metrica, amostragem=amostragem, comparacao=comparacao,
                estimativa=estimativa, ic95_inf=lo, ic95_sup=hi, p_bilateral_aprox=p_bil,
                n_replicas=len(reps))


def resumir(metrica, amostragem, reps, est):
    linhas = []
    for (cen, mod), v in reps.items():                       # intervalos marginais
        linhas.append(_linha(metrica, amostragem, f'{mod} (Cenário {cen})', est[(cen, mod)], v, p=False))
    for mod in MODELOS:                                      # efeito do vazamento
        linhas.append(_linha(metrica, amostragem, f'A - B: {mod}',
                             est[('A', mod)] - est[('B', mod)],
                             reps[('A', mod)] - reps[('B', mod)]))
    for cen in ('A', 'B'):                                   # comparação entre modelos
        for mod in ('Random Forest', 'Reg. Logística'):
            linhas.append(_linha(metrica, amostragem, f'{mod} - XGBoost (Cenário {cen})',
                                 est[(cen, mod)] - est[(cen, 'XGBoost')],
                                 reps[(cen, mod)] - reps[(cen, 'XGBoost')]))
    return linhas


# ==========================================
# 3. DIAGNÓSTICOS AUXILIARES
# ==========================================
def diagnosticos(df_a):
    treino, teste = particoes(df_a)
    print("\n=== Contagem de atrasos nula no Cenário A ===")
    for nome, sub in (('treino', treino), ('teste', teste)):
        zero = sub['SELLER_DELAY_COUNT'].fillna(0).to_numpy() == 0
        pos = sub[TARGET].astype(int).to_numpy() == 1
        print(f"[{nome}] positivos com SELLER_DELAY_COUNT=0: {int((zero & pos).sum())} de {int(pos.sum())}; "
              f"negativos com contagem nula: {int((zero & ~pos).sum())} "
              f"({100 * (zero & ~pos).sum() / (~pos).sum():.1f}% dos negativos)")

    mensal = (df_a.assign(mes=pd.to_datetime(df_a[COL_DATA]).dt.to_period('M').astype(str))
                  .groupby('mes')[TARGET].agg(pedidos='size', taxa_atraso='mean').reset_index())
    print("\n=== Taxa de atraso por mês de compra ===")
    print(mensal.round(4).to_string(index=False))
    mensal.to_csv('assets/taxa_atraso_mensal.csv', index=False)


# ==========================================
# EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    ctx = conectar_snowflake()
    print("Baixando datasets...")
    df_a = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_A_LEAKAGE")
    df_b = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_B_POINT_IN_TIME")
    os.makedirs('assets', exist_ok=True)

    res = {'A': treinar_e_avaliar_completo(df_a, "CENÁRIO A (Com Vazamento)"),
           'B': treinar_e_avaliar_completo(df_b, "CENÁRIO B (Point-in-Time)")}

    _, teste_a = particoes(df_a)
    _, teste_b = particoes(df_b)
    y = np.asarray(res['A']['XGBoost']['y_test'], dtype=int)
    # O pareamento só faz sentido se os dois cenários tiverem os mesmos registros, na mesma ordem
    assert np.array_equal(y, np.asarray(res['B']['XGBoost']['y_test'], dtype=int))
    assert teste_a[CHAVE].equals(teste_b[CHAVE]), "Conjuntos de teste diferentes entre A e B"

    probs = {(c, m): np.asarray(res[c][m]['y_prob']) for c in ('A', 'B') for m in MODELOS}
    est_auc = {k: roc_auc_score(y, p) for k, p in probs.items()}
    est_ap = {k: average_precision_score(y, p) for k, p in probs.items()}

    linhas = []
    for nome, amostrar in (('linhas', amostrador_linhas(len(y))),
                           ('vendedores', amostrador_vendedores(teste_a['SELLER_ID']))):
        print(f"\nBootstrap por {nome} ({N_BOOT} réplicas)...")
        auc, ap = bootstrap(y, probs, amostrar)
        linhas += resumir('ROC-AUC', nome, auc, est_auc)
        linhas += resumir('PR-AUC', nome, ap, est_ap)

    tab = pd.DataFrame(linhas)
    tab.to_csv('assets/bootstrap_pareado.csv', index=False)
    pd.set_option('display.width', 200)
    print(tab.round(4).to_string(index=False))

    diagnosticos(df_a)
    print("\nFinalizado! CSVs salvos em assets/")
