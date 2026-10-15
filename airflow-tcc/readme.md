# 1. Entre na pasta do seu projeto (caso já não esteja nela)
cd ~/github/tcc-1/

# 2. Ative o seu ambiente virtual
source venv_tcc/bin/activate

# 3. Diga ao Airflow para criar os arquivos dentro da sua pasta do TCC (e não bagunçar seu computador)
export AIRFLOW_HOME=~/github/tcc-1/airflow

# 4. Instale o Apache Airflow
pip install "apache-airflow==2.10.2"

# 5. inicializei o banco de dados do airflow
airflow db migrate

# 6. criar usuario admin
airflow users create \
    --username admin \
    --firstname Poliana \
    --lastname TCC \
    --role Admin \
    --email poli@ufrn.br \
    --password admin

# 7. Na Aba 1 (Para a Interface Web):
cd ~/github/tcc-1/
source venv_tcc/bin/activate
export AIRFLOW_HOME=~/github/tcc-1/airflow-tcc/
airflow webserver --port 8080

# 8. Na Aba 2 (Para o Orquestrador/Scheduler):
cd ~/github/tcc-1/
source venv_tcc/bin/activate
export AIRFLOW_HOME=~/github/tcc-1/airflow-tcc/
airflow scheduler

# 9. criar a pasta de dags e o arquivo
mkdir ~/github/tcc-1/airflow-tcc/dags
touch ~/github/tcc-1/airflow-tcc/dags/dag_tcc.py