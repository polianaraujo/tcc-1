"""
Gera os gráficos (PNG, 300 dpi) usados na monografia, no mesmo estilo visual
dos gráficos do script_ml.py (fonte e cores padrão do matplotlib):

  1. assets/ic95_roc_auc.png      -> ROC-AUC média com IC 95% (bootstrap), Cenários A e B
  2. assets/ablacao_xgboost.png   -> Ablação de variáveis do XGBoost (lê assets/ablacao_xgboost.csv)
  3. assets/benchmark_joins.png   -> Tempo das estratégias de junção (lê assets/benchmark_joins_execucoes.csv)

Uso (na pasta do projeto, com o venv ativo):
    python gerar_graficos_tcc.py

Os PNGs devem ser copiados para a pasta figuras/ do Overleaf.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Mesmas cores dos gráficos do script_ml.py (paleta padrão do matplotlib)
COR_A = "#1f77b4"   # Cenário A (com vazamento)
COR_B = "#ff7f0e"   # Cenário B (point-in-time)
COR_ACASO = "red"
DPI = 300
os.makedirs("assets", exist_ok=True)




# ==========================================================================
# 1. ROC-AUC com IC 95% (valores do bootstrap impressos pelo script_ml.py)
#    Formato: (AUC média, limite inferior, limite superior)
# ==========================================================================
def grafico_ic95():
    modelos = ["Reg. Logística", "Random Forest", "XGBoost"]
    cenario_a = [(0.5821, 0.5605, 0.6024), (0.6765, 0.6588, 0.6951), (0.6778, 0.6593, 0.6970)]
    cenario_b = [(0.4936, 0.4713, 0.5146), (0.5110, 0.4886, 0.5317), (0.5124, 0.4900, 0.5352)]

    x = np.arange(len(modelos))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    for dados, desloc, cor, rotulo in [
        (cenario_a, -w / 2, COR_A, "Cenário A (Com Vazamento)"),
        (cenario_b, +w / 2, COR_B, "Cenário B (Point-in-Time)"),
    ]:
        media = np.array([d[0] for d in dados])
        erro_inf = media - np.array([d[1] for d in dados])   # barras de erro assimétricas
        erro_sup = np.array([d[2] for d in dados]) - media
        ax.bar(x + desloc, media, w, color=cor, label=rotulo,
               yerr=[erro_inf, erro_sup], capsize=5)
        for xi, (m, _, sup) in zip(x + desloc, dados):
            ax.text(xi, sup + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(0.5, color=COR_ACASO, ls="--", lw=2, label="Acaso (0.50)")
    ax.set_title("Impacto do Point-in-Time Correctness na Performance dos Modelos (IC 95%)", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=11)
    ax.set_ylim(0.45, 0.73)
    ax.set_ylabel("ROC-AUC Média", fontsize=12)
    ax.grid(axis="y", ls="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("assets/ic95_roc_auc.png", dpi=DPI)
    plt.close(fig)
    print("Salvo: assets/ic95_roc_auc.png")


# ==========================================================================
# 2. Ablação de variáveis (lê o CSV gerado pelo analise_vazamento.py)
# ==========================================================================
def _ler_ic(texto):
    """Converte '[0.649, 0.680]' em (0.649, 0.680)."""
    lo, hi = re.findall(r"[-\d.]+", texto)
    return float(lo), float(hi)


def _rotulo_curto(cfg):
    return (cfg.replace("Todas as variáveis", "Todas")
               .replace("Somente ", "Só ")
               .replace("SELLER_ORDER_VOLUME", "VOLUME")
               .replace("SELLER_DELAY_COUNT", "DELAY_COUNT")
               .replace("SELLER_AVG_REVIEW_SCORE", "AVG_REVIEW"))


def grafico_ablacao(caminho="assets/ablacao_xgboost.csv"):
    df = pd.read_csv(caminho)
    rotulos = [_rotulo_curto(c) for c in df["configuracao"]]
    y = np.arange(len(df))[::-1]          # primeira configuração no topo
    h = 0.36
    base = 0.44                           # barras começam em 0,44 para dar zoom

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for cen, desloc, cor, nome in [("A", +h / 2, COR_A, "Cenário A (Com Vazamento)"),
                                   ("B", -h / 2, COR_B, "Cenário B (Point-in-Time)")]:
        auc = df[f"AUC_{cen}"].to_numpy()
        ics = np.array([_ler_ic(t) for t in df[f"IC95_{cen}"]])
        ax.barh(y + desloc, auc - base, h, left=base, color=cor, label=nome,
                xerr=[auc - ics[:, 0], ics[:, 1] - auc], capsize=4)

    ax.axvline(0.5, color=COR_ACASO, ls="--", lw=2, label="Acaso (0.50)")
    ax.set_title("Ablação de Variáveis: ROC-AUC do XGBoost (IC 95%)", fontsize=14, pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(rotulos, fontsize=11)
    ax.set_xlim(base, 0.73)
    ax.set_xlabel("ROC-AUC no Conjunto de Teste", fontsize=12)
    ax.grid(axis="x", ls="--", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("assets/ablacao_xgboost.png", dpi=DPI)
    plt.close(fig)
    print("Salvo: assets/ablacao_xgboost.png")


# ==========================================================================
# 3. Benchmark das junções (lê o CSV gerado pelo benchmark_joins.py)
#    SQL: tempo de execução no Snowflake (QUERY_HISTORY)
#    pandas: tempo total no cliente (extração + junção)
#    Mediana das repetições, descartando a 1ª (aquecimento)
# ==========================================================================
def grafico_benchmark(caminho="assets/benchmark_joins_execucoes.csv"):
    df = pd.read_csv(caminho)
    df = df[(df["status"] == "ok") & (df["rep"] > 1)].copy()
    df["tempo_s"] = np.where(df["estrategia"] == "pandas.merge_asof",
                             df["parede_ms"], df["execucao_ms"]) / 1000
    med = df.groupby(["estrategia", "escala"])["tempo_s"].median().unstack("escala")

    ordem = ["ASOF JOIN", "Subconsulta correlacionada", "Non-equi join + QUALIFY", "pandas.merge_asof"]
    nomes = ["ASOF JOIN", "Subconsulta\ncorrelacionada", "Non-equi join\n+ QUALIFY",
             "pandas.merge_asof\n(extração + junção)"]
    med = med.reindex(ordem)
    escalas = sorted(med.columns)
    linhas = df.groupby("escala")["linhas"].first()
    tons = ["#aec7e8", COR_A, "#0b3d66"]   # azul claro / azul padrão / azul escuro

    x = np.arange(len(ordem))
    w = 0.72 / len(escalas)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, esc in enumerate(escalas):
        desloc = (i - (len(escalas) - 1) / 2) * w
        rotulo = f"Escala {esc}x ({linhas[esc]:,} linhas)".replace(",", ".")
        barras = ax.bar(x + desloc, med[esc], w, color=tons[i % len(tons)], label=rotulo)
        for b in barras:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v * 1.12,
                    f"{v:.2f} s", ha="center", va="bottom", fontsize=10)

    ax.set_yscale("log")
    ax.set_ylim(med.min().min() * 0.4, med.max().max() * 4.5)
    ax.set_title("Tempo de Execução das Estratégias de Junção Point-in-Time", fontsize=14, pad=15)
    ax.set_ylabel("Tempo (s, escala log)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", ls="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("assets/benchmark_joins.png", dpi=DPI)
    plt.close(fig)
    print("Salvo: assets/benchmark_joins.png")
    print(med.round(3).to_string())


if __name__ == "__main__":
    grafico_ic95()
    grafico_ablacao()
    grafico_benchmark()
