# Databricks notebook source
# MAGIC %pip install PyYAML
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml
import json
import requests

# COMMAND ----------

# get user id for who is running notebook:
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# create cleaned id to serve as prefix in feature branch builds
trunc_user = user.split('@')[0].replace('.', '_') + "_"
print(f'User-Based Prefix: {trunc_user}')

# COMMAND ----------

# MAIN branch defaults for config
raw_config = f'''

user: {trunc_user}
host_url: infinitive-sandbox2.cloud.databricks.com
catalog: main
dbName: intern_project_2024

'''

yaml_config = yaml.safe_load(raw_config)

# COMMAND ----------

host_url = yaml_config['host_url']
catalog = yaml_config['catalog']
dbName = yaml_config['dbName']

def setup_catalog_and_db():
    spark.sql(f"USE CATALOG `{catalog}`")
    spark.sql(f"""create database if not exists `{dbName}` """)
    spark.sql(f"USE DATABASE `{dbName}`")

setup_catalog_and_db()

# COMMAND ----------

# hardcoded for now, this shouldn't change unless a new prinicpal is created
def grant_service_principal_schema_usage():
    spark.sql('GRANT USAGE ON CATALOG `main` TO `5386a7e1-5db0-47df-92a9-929a7bb4e18b`');
    spark.sql('GRANT USAGE ON DATABASE `main`.`intern_project_2024` TO `5386a7e1-5db0-47df-92a9-929a7bb4e18b`');

def grant_user_schema_usage():
    spark.sql(f'GRANT USAGE ON CATALOG `main` TO `{user}`');
    spark.sql(f'GRANT USAGE ON DATABASE `main`.`intern_project_2024` TO `{user}`');

# COMMAND ----------


