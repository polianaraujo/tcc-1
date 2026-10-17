"""
Avaliação cruzada entre cenários (treino x teste), para medir o training-serving skew.

  A -> A : treino e teste com as variáveis vazadas (experimento principal)
  A -> B : modelo treinado em A e avaliado sobre as variáveis do Cenário B. Reproduz a operação
           de um pipeline ingênuo: treino com vazamento, produção com variáveis calculadas
           apenas com a informação conhecida no instante da compra.
  B -> B : experimento principal (treino e teste point-in-time)
  B -> A : diagnóstico complementar (modelo treinado em B, avaliado sobre as variáveis de A)

Mesmos hiperparâmetros, sementes e partição temporal do script_ml.py. Os mesmos registros de teste
são usados nos dois cenários, e o bootstrap usa os mesmos índices em todas as combinações
(pareado). IC 95% por reamostragem de registros e de vendedores.

Uso (na pasta do projeto, com o venv ativo):
    python avaliacao_cruzada.py

Saída: assets/avaliacao_cruzada.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from script_ml import conectar_snowflake, extrair_dados
from bootstrap_pareado import (particoes, bootstrap, amostrador_linhas, amostrador_vendedores,
                               _linha, MODELOS, CHAVE, TARGET)

FEATURES = ['SELLER_ORDER_VOLUME', 'SELLER_DELAY_COUNT', 'SELLER_AVG_REVIEW_SCORE']

# Diferenças de interesse: (rótulo, configuração 1, configuração 2)
DIFERENCAS = [
    ('A->A - A->B (queda ao servir o modelo de A com variáveis de B)', 'A->A', 'A->B'),
    ('A->B - B->B (treino em A contra treino em B, ambos testados em B)', 'A->B', 'B->B'),
]


def criar_modelos(y_treino):
    # Mesmos hiperparâmetros de script_ml.treinar_e_avaliar_completo
    ratio = (y_treino == 0).sum() / max((y_treino == 1).sum(), 1)
    return {
        "Random Forest": RandomForestClassifier(random_state=42, max_depth=5, n_jobs=-1, class_weight='balanced'),
        "Reg. Logística": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, max_depth=5, scale_pos_weight=ratio, eval_metric='logloss'),
    }


def preparar(df):
    treino, teste = particoes(df)
    x = lambda d: d[FEATURES].astype(float).fillna(0)
    return x(treino), treino[TARGET].astype(int).to_numpy(), x(teste), teste[TARGET].astype(int).to_numpy(), teste


def resumir(metrica, amostragem, reps, est):
    linhas = []
    for (cfg, mod), v in reps.items():
        linhas.append(_linha(metrica, amostragem, f'{mod} ({cfg})', est[(cfg, mod)], v, p=False))
    for mod in MODELOS:
        for rotulo, c1, c2 in DIFERENCAS:
            linhas.append(_linha(metrica, amostragem, f'{mod}: {rotulo}',
                                 est[(c1, mod)] - est[(c2, mod)],
                                 reps[(c1, mod)] - reps[(c2, mod)]))
    return linhas


if __name__ == "__main__":
    ctx = conectar_snowflake()
    print("Baixando datasets...")
    df_a = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_A_LEAKAGE")
    df_b = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_B_POINT_IN_TIME")
    os.makedirs('assets', exist_ok=True)

    dados = {'A': preparar(df_a), 'B': preparar(df_b)}
    y = dados['A'][3]
    # O pareamento exige os mesmos registros de teste, na mesma ordem, nos dois cenários
    assert np.array_equal(y, dados['B'][3])
    assert dados['A'][4][CHAVE].equals(dados['B'][4][CHAVE]), "Conjuntos de teste diferentes entre A e B"

    probs = {}
    for treino_c in ('A', 'B'):
        X_tr, y_tr = dados[treino_c][0], dados[treino_c][1]
        for nome, modelo in criar_modelos(y_tr).items():
            modelo.fit(X_tr, y_tr)
            for teste_c in ('A', 'B'):
                probs[(f'{treino_c}->{teste_c}', nome)] = modelo.predict_proba(dados[teste_c][2])[:, 1]
            print(f"  treinado: {nome} em {treino_c}")

    est_auc = {k: roc_auc_score(y, p) for k, p in probs.items()}
    est_ap = {k: average_precision_score(y, p) for k, p in probs.items()}

    linhas = []
    for nome, amostrar in (('linhas', amostrador_linhas(len(y))),
                           ('vendedores', amostrador_vendedores(dados['A'][4]['SELLER_ID']))):
        print(f"\nBootstrap por {nome}...")
        auc, ap = bootstrap(y, probs, amostrar)
        linhas += resumir('ROC-AUC', nome, auc, est_auc)
        linhas += resumir('PR-AUC', nome, ap, est_ap)

    tab = pd.DataFrame(linhas)
    tab.to_csv('assets/avaliacao_cruzada.csv', index=False)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 90)
    print(tab.round(4).to_string(index=False))
    print("\nFinalizado! CSV salvo em assets/avaliacao_cruzada.csv")
