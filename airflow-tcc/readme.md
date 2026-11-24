# Airflow (orquestração)

A DAG `pipeline_tcc_dbt_xgboost` fica em `dags/dag_tcc.py` e não tem agendamento (`schedule=None`): é disparada manualmente.

```
executar_modelos_dbt  ──►  treinar_modelo_xgboost
      (dbt run)       └─►  analisar_vazamento
```

As análises estatísticas complementares (`bootstrap_pareado.py`, `avaliacao_cruzada.py`, `ablacao_pareada.py`, `purga_resguardo.py`, `benchmark_joins.py`) e os gráficos rodam fora da DAG. Ver o `README.md` da raiz.

## Subir o Airflow localmente

Rode a partir da raiz do repositório. Use um terminal para o servidor web e outro para o agendador; nos dois, ative o ambiente e aponte o diretório do Airflow:

```bash
source venv_tcc/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow-tcc
```

Instalação (uma vez), com o arquivo de restrições oficial:

```bash
PYV=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
pip install "apache-airflow==2.10.2" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-${PYV}.txt"
```

Inicialização (uma vez):

```bash
airflow db migrate
airflow users create --username admin --role Admin \
  --firstname Admin --lastname TCC --email admin@example.com --password <senha>
```

Execução:

```bash
airflow webserver --port 8080     # terminal 1
airflow scheduler                 # terminal 2
```

Acesse http://localhost:8080, ative a DAG `pipeline_tcc_dbt_xgboost` e use *Trigger DAG*.

Para encerrar, pressione `Ctrl + C` em cada terminal (agendador e servidor web) e execute `deactivate`.

## Arquivos que não vão para o repositório

`airflow.cfg` (contém a chave secreta do servidor web), `airflow.db`, `logs/` e `webserver_config.py` são gerados localmente e estão no `.gitignore`. Não versione credenciais.
