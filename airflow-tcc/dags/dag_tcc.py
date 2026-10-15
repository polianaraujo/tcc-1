from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'poliana',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

with DAG(
    'pipeline_tcc_dbt_xgboost',
    default_args=default_args,
    description='Orquestração do TCC: Transformação e Análise Forense de Vazamento',
    schedule=None, # Atualizado para evitar warnings de depreciação
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['tcc', 'dbt', 'mlops'],
) as dag:

    rodar_dbt = BashOperator(
        task_id='executar_modelos_dbt',
        bash_command="""
        cd ~/github/tcc-1/tcc_feature_store && \
        source ~/github/tcc-1/venv_tcc/bin/activate && \
        dbt run
        """
    )

    treinar_modelo = BashOperator(
        task_id='treinar_modelo_xgboost',
        bash_command="""
        cd ~/github/tcc-1/ && \
        source ~/github/tcc-1/venv_tcc/bin/activate && \
        python script_ml.py
        """
    )

    # Nova tarefa de ablação e diff
    analisar_vazamento = BashOperator(
        task_id='analisar_vazamento',
        bash_command="""
        cd ~/github/tcc-1/ && \
        source ~/github/tcc-1/venv_tcc/bin/activate && \
        python analise_vazamento.py
        """
    )

    # O dbt roda primeiro. Depois, as duas análises rodam em paralelo
    rodar_dbt >> [treinar_modelo, analisar_vazamento]