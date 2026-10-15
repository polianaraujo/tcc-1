"""
Análises complementares ao experimento Cenário A (vazamento) vs Cenário B (point-in-time).

  1. Diferença A vs B pedido a pedido -> QUANTO do Dataset A foi contaminado.
  2. Ablação de variáveis (XGBoost)   -> ONDE (em qual variável) o vazamento estava.

Reaproveita a conexão e o bootstrap do script_ml.py (mesma pasta).
Saídas: assets/diff_A_vs_B.csv e assets/ablacao_xgboost.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from script_ml import conectar_snowflake, extrair_dados, bootstrap_auc_ci

FEATURES = ['SELLER_ORDER_VOLUME', 'SELLER_DELAY_COUNT', 'SELLER_AVG_REVIEW_SCORE']
TARGET = 'TARGET_HOUVE_ATRASO'
COL_DATA = 'DATA_DO_PEDIDO'

# Chave que identifica UMA linha nos dois datasets Gold (granularidade pedido-vendedor).
# Pedidos com itens de mais de um vendedor aparecem uma vez para cada vendedor.
CHAVE = ['ORDER_ID', 'SELLER_ID']


def validar_chave(df, nome):
    faltando = [c for c in CHAVE + [COL_DATA] if c not in df.columns]
    if faltando:
        raise KeyError(f"[{nome}] colunas ausentes: {faltando}. Disponíveis: {df.columns.tolist()}")
    
    dup = df.duplicated(CHAVE).sum()
    if dup:
        # Com a chave (ORDER_ID, SELLER_ID) não deve haver repetição: se houver, é erro nos modelos dbt
        raise ValueError(f"[{nome}] {dup} linhas repetidas na chave {CHAVE}. Verifique os modelos Gold.")
    print(f"[{nome}] {len(df)} linhas, chave {CHAVE} única.")


# ==========================================
# 1. DIFERENÇA A vs B PEDIDO A PEDIDO
# ==========================================
def comparar_datasets(df_a, df_b, features=FEATURES, frac_teste=0.2):
    validar_chave(df_a, 'A')
    validar_chave(df_b, 'B')

    m = df_a[CHAVE + [COL_DATA] + features].merge(
        df_b[CHAVE + features], on=CHAVE, how='outer', suffixes=('_A', '_B'), indicator=True
    )
    print("Cobertura do join (both = pedidos presentes em A e B):", m['_merge'].value_counts().to_dict())
    m = m[m['_merge'] == 'both'].copy()

    # Janela de teste = mesmo critério do treino (últimos 20% no tempo)
    datas = m[COL_DATA].sort_values().reset_index(drop=True)
    corte = datas.iloc[int(len(datas) * (1 - frac_teste))]
    janelas = {
        'todo_periodo': m,
        'janela_teste': m[m[COL_DATA] >= corte],
    }

    linhas = []
    for nome_j, sub in janelas.items():
        alguma_diferente = np.zeros(len(sub), dtype=bool)
        for f in features:
            a = sub[f + '_A'].astype(float).to_numpy()
            b = sub[f + '_B'].astype(float).to_numpy()
            diferente = ~np.isclose(a, b, equal_nan=True)   # NaN vs valor conta como diferente
            alguma_diferente |= diferente
            ambos = ~np.isnan(a) & ~np.isnan(b)
            linhas.append({
                'janela': nome_j,
                'variavel': f,
                'n_linhas': len(sub),
                'pct_linhas_dif': 100 * diferente.mean(),
                'pct_A_maior': 100 * np.mean(a > b),
                'pct_A_menor': 100 * np.mean(a < b),
                'media_A': np.nanmean(a),
                'media_B': np.nanmean(b),
                'desvio_abs_medio': np.mean(np.abs(a[ambos] - b[ambos])) if ambos.any() else np.nan,
                'pct_nulo_A': 100 * np.isnan(a).mean(),
                'pct_nulo_B': 100 * np.isnan(b).mean(),   # vendedor sem histórico no instante do pedido
            })
        print(f"[{nome_j}] {100 * alguma_diferente.mean():.1f}% dos pedidos têm ao menos uma "
              f"variável diferente entre A e B")
    return pd.DataFrame(linhas)


# ==========================================
# 2. ABLAÇÃO DE VARIÁVEIS (XGBoost)
# ==========================================
def _preparar(df, cols):
    # Desempate pela CHAVE: sem isso a ordem de linhas com a mesma data varia entre execuções
    df = df.sort_values([COL_DATA] + CHAVE).reset_index(drop=True)
    X = df[cols].astype(float).fillna(0)
    y = df[TARGET].astype(int)
    corte = int(len(df) * 0.8)
    return X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]


def ablacao(df_a, df_b, features=FEATURES, n_boot=1000):
    configs = {'Todas as variáveis': features}
    for f in features:
        configs[f'Sem {f}'] = [x for x in features if x != f]
    for f in features:
        configs[f'Somente {f}'] = [f]

    linhas = []
    for nome_cfg, cols in configs.items():
        linha = {'configuracao': nome_cfg}
        for cenario, df in [('A', df_a), ('B', df_b)]:
            X_tr, X_te, y_tr, y_te = _preparar(df, cols)
            ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            modelo = XGBClassifier(random_state=42, max_depth=5,
                                   scale_pos_weight=ratio, eval_metric='logloss')
            modelo.fit(X_tr, y_tr)
            prob = modelo.predict_proba(X_te)[:, 1]
            _, lo, hi = bootstrap_auc_ci(y_te.values, prob, n_boot=n_boot)
            linha[f'AUC_{cenario}'] = roc_auc_score(y_te, prob)
            linha[f'IC95_{cenario}'] = f'[{lo:.3f}, {hi:.3f}]'
        linha['delta_A_menos_B'] = linha['AUC_A'] - linha['AUC_B']
        linhas.append(linha)
        print(f"  {nome_cfg:<38} AUC A={linha['AUC_A']:.4f} | AUC B={linha['AUC_B']:.4f}")
    return pd.DataFrame(linhas)


# ==========================================
# EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    ctx = conectar_snowflake()
    print("Baixando datasets...")
    df_a = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_A_LEAKAGE")
    df_b = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_B_POINT_IN_TIME")
    os.makedirs('assets', exist_ok=True)

    print("\n=== 1) DIFERENÇA A vs B PEDIDO A PEDIDO ===")
    tab_diff = comparar_datasets(df_a, df_b)
    print(tab_diff.round(3).to_string(index=False))
    tab_diff.to_csv('assets/diff_A_vs_B.csv', index=False)

    print("\n=== 2) ABLAÇÃO DE VARIÁVEIS (XGBoost) ===")
    tab_abl = ablacao(df_a, df_b)
    print(tab_abl.round(4).to_string(index=False))
    tab_abl.to_csv('assets/ablacao_xgboost.csv', index=False)

    print("\nAnálises finalizadas! CSVs salvos em assets/")