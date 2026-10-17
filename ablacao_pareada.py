"""
Ablação com comparações pareadas e diagnóstico de não estacionariedade.

  1. Reexecuta a ablação do XGBoost (mesmos hiperparâmetros, partição e sementes de analise_vazamento.py).
  2. IC 95% pareado (mesmos índices de reamostragem em todas as configurações) para:
       - cada configuração menos 'Todas as variáveis', em cada cenário;
       - A - B, para cada configuração.
     Reamostragem de registros e de vendedores (conglomerados).
  3. AUC univariada de cada variável bruta, no treino e no teste, em cada cenário (sem modelo).
     Valores acima de 0,5 indicam que valores maiores da variável se associam a mais atrasos; se o
     sentido muda entre treino e teste, a relação é não estacionária.

Uso (na pasta do projeto, com o venv ativo):
    python ablacao_pareada.py

Saídas: assets/ablacao_pareada.csv e assets/auc_univariada.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from script_ml import conectar_snowflake, extrair_dados
from analise_vazamento import FEATURES, _preparar
from bootstrap_pareado import (particoes, bootstrap, amostrador_linhas, amostrador_vendedores,
                               _linha, CHAVE, TARGET)

TODAS = 'Todas as variáveis'


def configuracoes(features=FEATURES):
    cfg = {TODAS: features}
    for f in features:
        cfg[f'Sem {f}'] = [x for x in features if x != f]
    for f in features:
        cfg[f'Somente {f}'] = [f]
    return cfg


def prever(df, cols):
    X_tr, X_te, y_tr, y_te = _preparar(df, cols)
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    modelo = XGBClassifier(random_state=42, max_depth=5, scale_pos_weight=ratio, eval_metric='logloss')
    modelo.fit(X_tr, y_tr)
    return y_te.to_numpy(dtype=int), modelo.predict_proba(X_te)[:, 1]


def resumir(amostragem, reps, est, nomes):
    linhas = []
    for (cen, cfg), v in reps.items():                       # intervalos marginais
        linhas.append(_linha('ROC-AUC', amostragem, f'{cfg} (Cenário {cen})', est[(cen, cfg)], v, p=False))
    for cfg in nomes:                                        # A - B por configuração
        linhas.append(_linha('ROC-AUC', amostragem, f'A - B: {cfg}',
                             est[('A', cfg)] - est[('B', cfg)], reps[('A', cfg)] - reps[('B', cfg)]))
    for cen in ('A', 'B'):                                   # configuração - todas as variáveis
        for cfg in nomes:
            if cfg == TODAS:
                continue
            linhas.append(_linha('ROC-AUC', amostragem, f'{cfg} - Todas (Cenário {cen})',
                                 est[(cen, cfg)] - est[(cen, TODAS)],
                                 reps[(cen, cfg)] - reps[(cen, TODAS)]))
    return linhas


def auc_univariada(df, cenario):
    treino, teste = particoes(df)
    linhas = []
    for f in FEATURES:
        for periodo, sub in (('treino', treino), ('teste', teste)):
            y = sub[TARGET].astype(int).to_numpy()
            x = sub[f].astype(float).fillna(0).to_numpy()
            linhas.append(dict(cenario=cenario, variavel=f, periodo=periodo,
                               n=len(y), positivos=int(y.sum()), auc=roc_auc_score(y, x)))
    return linhas


if __name__ == "__main__":
    ctx = conectar_snowflake()
    print("Baixando datasets...")
    df_a = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_A_LEAKAGE")
    df_b = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_B_POINT_IN_TIME")
    os.makedirs('assets', exist_ok=True)

    cfgs = configuracoes()
    probs, y = {}, None
    for cen, df in (('A', df_a), ('B', df_b)):
        for nome, cols in cfgs.items():
            y_c, p = prever(df, cols)
            # O pareamento exige os mesmos registros de teste, na mesma ordem
            assert y is None or np.array_equal(y, y_c)
            y = y_c
            probs[(cen, nome)] = p
            print(f"  {cen} | {nome}")

    _, teste_a = particoes(df_a)
    _, teste_b = particoes(df_b)
    assert teste_a[CHAVE].equals(teste_b[CHAVE]), "Conjuntos de teste diferentes entre A e B"
    est = {k: roc_auc_score(y, p) for k, p in probs.items()}

    linhas = []
    for nome, amostrar in (('linhas', amostrador_linhas(len(y))),
                           ('vendedores', amostrador_vendedores(teste_a['SELLER_ID']))):
        print(f"\nBootstrap por {nome}...")
        auc, _ = bootstrap(y, probs, amostrar)
        linhas += resumir(nome, auc, est, list(cfgs))

    tab = pd.DataFrame(linhas)
    tab.to_csv('assets/ablacao_pareada.csv', index=False)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 80)
    print(tab.round(4).to_string(index=False))

    uni = pd.DataFrame(auc_univariada(df_a, 'A') + auc_univariada(df_b, 'B'))
    uni.to_csv('assets/auc_univariada.csv', index=False)
    print("\n=== AUC univariada (variável bruta, sem modelo) ===")
    print(uni.round(4).to_string(index=False))
    print("\nFinalizado! CSVs salvos em assets/")
