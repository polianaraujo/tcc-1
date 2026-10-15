import snowflake.connector
import pandas as pd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import os
import numpy as np
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, average_precision_score, confusion_matrix, classification_report
import warnings

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# Oculta os alertas da regressão logística e outros
warnings.filterwarnings('ignore')

# ==========================================
# 1. FUNÇÃO DE CONEXÃO E EXTRAÇÃO
# ==========================================
def conectar_snowflake():
    load_dotenv(override=True)
    
    # 1. Carregar a chave privada do arquivo rsa_key.p8
    with open("rsa_key.p8", "rb") as key:
        p_key = serialization.load_pem_private_key(
            key.read(),
            password=None,
            backend=default_backend()
        )

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # 2. Configurar a conexão com o Snowflake usando a chave em vez de senha
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        private_key=pkb,
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )
    print("Conexão estabelecida com sucesso usando chave RSA!")
    return ctx

def extrair_dados(ctx, query):
    cur = ctx.cursor()
    cur.execute(query)
    return cur.fetch_pandas_all()

# ==========================================
# 2. FUNÇÃO DE TREINAMENTO E AVALIAÇÃO
# ==========================================
def treinar_e_avaliar_completo(df, nome_cenario):
    # Desempate determinístico: (pedido, vendedor) identifica uma linha de forma única
    df = df.sort_values(['DATA_DO_PEDIDO', 'ORDER_ID', 'SELLER_ID']).reset_index(drop=True)
    features = ['SELLER_ORDER_VOLUME', 'SELLER_DELAY_COUNT', 'SELLER_AVG_REVIEW_SCORE']
    target = 'TARGET_HOUVE_ATRASO'
    df[features] = df[features].fillna(0)
    X, y = df[features], df[target]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Prevenir erro caso y_train tenha apenas uma classe (raro, mas bom para scripts automáticos)
    if (y_train == 1).sum() == 0:
        ratio = 1
    else:
        ratio = (y_train == 0).sum() / (y_train == 1).sum()

    modelos = {
        "Random Forest": RandomForestClassifier(random_state=42, max_depth=5, n_jobs=-1, class_weight='balanced'),
        "Reg. Logística": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, max_depth=5, scale_pos_weight=ratio, eval_metric='logloss'),
    }

    resultados = {}
    print(f"=== {nome_cenario} ===")
    for nome, model in modelos.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        baseline_pr = y_test.mean()  # PR-AUC de um classificador aleatório = taxa de positivos na base
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        print(f"\n{nome}")
        print(f"  ROC-AUC : {auc:.4f}")
        print(f"  PR-AUC  : {pr_auc:.4f}  (baseline aleatório = {baseline_pr:.4f})")
        print(f"  Matriz de confusão -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))

        # resultados[nome] = {"y_test": y_test.values, "y_prob": y_prob}
        resultados[nome] = {"y_test": y_test.values, "y_prob": y_prob, "modelo": model}
    print("-" * 65)
    return resultados

# ==========================================
# 3. FUNÇÃO DE INTERVALO DE CONFIANÇA
# ==========================================
def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi

# ==========================================
# 4. BLOCO DE EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("Iniciando pipeline de Machine Learning orquestrada pelo Airflow...")
    
    # 1. Conexão
    ctx = conectar_snowflake()
    
    # 2. Extração
    print("Baixando Dataset A (Leakage)...")
    df_a_leakage = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_A_LEAKAGE")
    
    print("Baixando Dataset B (Point-in-Time)...")
    df_b_point_in_time = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_B_POINT_IN_TIME")
    
    print("Dados carregados com sucesso!\n")
    
    # 3. Treinamento e Avaliação Completa
    resultados_a = treinar_e_avaliar_completo(df_a_leakage, "CENÁRIO A (Com Vazamento)")
    resultados_b = treinar_e_avaliar_completo(df_b_point_in_time, "CENÁRIO B (Point-in-Time)")
    
    # 4. Cálculo dos Intervalos de Confiança
    print("Calculando Intervalos de Confiança (Bootstrap)...")
    for nome, cenario_dict in [("A", resultados_a), ("B", resultados_b)]:
        print(f"\n=== Cenário {nome} ===")
        for modelo, r in cenario_dict.items():
            media, lo, hi = bootstrap_auc_ci(r["y_test"], r["y_prob"])
            print(f"  {modelo:<17} AUC médio={media:.4f}  IC 95% [{lo:.4f}, {hi:.4f}]")
    
    print("\nGerando gráficos de performance...")
    os.makedirs('assets', exist_ok=True)
    
    y_verdadeiro = resultados_a["XGBoost"]["y_test"] 
    probs_xgb_cenario_a = resultados_a["XGBoost"]["y_prob"]
    probs_xgb_cenario_b = resultados_b["XGBoost"]["y_prob"]

    # O gráfico de barras com IC 95% é gerado por gerar_graficos_tcc.py (assets/ic95_roc_auc.png)

    # --- GRÁFICO 2: CURVAS ROC E PR-AUC (ATUALIZADO) ---
    print("Gerando curvas ROC e PR-AUC...")
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    fpr_a, tpr_a, _ = roc_curve(y_verdadeiro, probs_xgb_cenario_a)
    fpr_b, tpr_b, _ = roc_curve(y_verdadeiro, probs_xgb_cenario_b)
    
    # Calculando os valores exatos de AUC para as legendas
    auc_roc_a = roc_auc_score(y_verdadeiro, probs_xgb_cenario_a)
    auc_roc_b = roc_auc_score(y_verdadeiro, probs_xgb_cenario_b)

    ax1.plot(fpr_a, tpr_a, color='#1f77b4', linewidth=2, label=f'Cenário A (Vazamento) - AUC: {auc_roc_a:.3f}')
    ax1.plot(fpr_b, tpr_b, color='#ff7f0e', linewidth=2, label=f'Cenário B (Point-in-Time) - AUC: {auc_roc_b:.3f}')
    ax1.plot([0, 1], [0, 1], color='red', linestyle='--', label='Acaso')

    ax1.set_title('Colapso da Curva ROC (XGBoost)', fontsize=14)
    ax1.set_xlabel('Taxa de Falsos Positivos')
    ax1.set_ylabel('Taxa de Verdadeiros Positivos')
    ax1.legend()
    ax1.grid(alpha=0.3)

    prec_a, rec_a, _ = precision_recall_curve(y_verdadeiro, probs_xgb_cenario_a)
    prec_b, rec_b, _ = precision_recall_curve(y_verdadeiro, probs_xgb_cenario_b)
    
    pr_auc_a = average_precision_score(y_verdadeiro, probs_xgb_cenario_a)
    pr_auc_b = average_precision_score(y_verdadeiro, probs_xgb_cenario_b)
    baseline_aleatorio = y_verdadeiro.mean() 

    ax2.plot(rec_a, prec_a, color='#1f77b4', linewidth=2, label=f'Cenário A - PR-AUC: {pr_auc_a:.3f}')
    ax2.plot(rec_b, prec_b, color='#ff7f0e', linewidth=2, label=f'Cenário B - PR-AUC: {pr_auc_b:.3f}')
    ax2.axhline(y=baseline_aleatorio, color='red', linestyle='--', label=f'Baseline ({baseline_aleatorio:.4f})')

    ax2.set_title('Colapso da Curva Precision-Recall (XGBoost)', fontsize=14)
    ax2.set_xlabel('Recall (Revocação)')
    ax2.set_ylabel('Precision (Precisão)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig_curves.tight_layout()
    fig_curves.savefig('assets/curvas_roc_pr_xgboost.png', dpi=300)
    print("Curvas salvas em assets/curvas_roc_pr_xgboost.png!")

    # --- GRÁFICO 3: FEATURE IMPORTANCE ("A ARMA DO CRIME") ---
    print("Gerando gráfico de Feature Importance...")
    fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
    
    features = ['Volume de Pedidos', 'Contagem de Atrasos', 'Score Médio de Avaliação']
    # Extraindo os pesos diretamente dos modelos XGBoost treinados
    importances_a = resultados_a["XGBoost"]["modelo"].feature_importances_
    importances_b = resultados_b["XGBoost"]["modelo"].feature_importances_
    
    y_pos = np.arange(len(features))
    width = 0.35
    
    ax_feat.barh(y_pos + width/2, importances_a, width, label='Cenário A (Com Vazamento)', color='#1f77b4')
    ax_feat.barh(y_pos - width/2, importances_b, width, label='Cenário B (Point-in-Time)', color='#ff7f0e')
    
    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels(features, fontsize=11)
    ax_feat.set_xlabel('Importância Relativa (Ganho)', fontsize=12)
    ax_feat.set_title('Deslocamento de Importância das Variáveis (XGBoost)', fontsize=14, pad=15)
    ax_feat.legend()
    ax_feat.grid(axis='x', linestyle='--', alpha=0.6)
    
    fig_feat.tight_layout()
    fig_feat.savefig('assets/feature_importance_xgboost.png', dpi=300)
    print("Gráfico de Feature Importance salvo em assets/feature_importance_xgboost.png!")
    
    print("\nPipeline de Machine Learning finalizada com sucesso!")