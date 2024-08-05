# Databricks notebook source
#import dependencies
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, FloatType, DateType, IntegerType
import pandas as pd

# COMMAND ----------

# MAGIC %run ./configuration

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;
# MAGIC -- # dbutils.fs.ls('/Volumes/')

# COMMAND ----------

print(yaml_config.keys())
print(yaml_config.values())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CREATE VOLUME IF NOT EXISTS intern_datasets;
# MAGIC SHOW VOLUMES;

# COMMAND ----------

# Do not need to run again!
# moves mounted folder into a volume
source_dir = 'dbfs:/FileStore/user/alexander.bonner@infinitive.com/credit_card_attrition/final_data'
dest_dir = 'dbfs:/Volumes/main/intern_project_2024/intern_datasets'

dbutils.fs.cp(source_dir, dest_dir, recurse=True)

# COMMAND ----------

dbutils.fs.ls("dbfs:/Volumes/main/intern_project_2024/intern_datasets")
agg_df = spark.read.csv('dbfs:/Volumes/main/intern_project_2024/intern_datasets/final_agg_dataset.csv/', header=True, inferSchema=True)
agg_df.show()

# COMMAND ----------

# schema for closed accounts table
closed_path = 'dbfs:/Volumes/main/intern_project_2024/intern_datasets/closed_accounts_table.csv/'
closed_schema = StructType([StructField('customer_id', StringType(), True),
                             StructField('status', StringType(), True), 
                             StructField('account_id', StringType(), True)]
                            )

closed_df = spark.read.csv(closed_path, schema=closed_schema)
closed_df.show()  
closed_pdf = closed_df.toPandas() 
closed_pdf.to_csv('closed_table_bronze.csv')

# COMMAND ----------

# create schema for credit account database table
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType
credit_account_path = 'dbfs:/Volumes/main/intern_project_2024/intern_datasets/credit_account_table.csv/'

credit_account_schema = StructType([StructField('account_id', StringType(), True),

                     StructField('customer_id', StringType(), True),

                     StructField('card_category', StringType(), True),

                     StructField('credit_limit', FloatType(), True),

                     StructField('account_open_date', DateType(), True)]

                    )

credit_account_df = spark.read.csv(credit_account_path, schema = credit_account_schema)
credit_account_df.show()

account_pdf = credit_account_df.toPandas() 
account_pdf.to_csv('credit_account_table_bronze.csv')

# COMMAND ----------

#schema for credit metrics table

#file path variable
metrics_path = "dbfs:/Volumes/main/intern_project_2024/intern_datasets/credit_metrics_table.csv/"

#defining schema for credit metrics table
metrics_schema = StructType([
    StructField("customer_id", StringType(), True), 
    StructField("total_relationship_count",FloatType(), True), 
    StructField("contacts_count_12mo", FloatType(), True), 
    StructField("total_revolving_bal", FloatType(), True), 
    StructField("total_amt_chng_Q4_Q1", FloatType(), True), 
    StructField("total_ct_chng_Q4_Q1​", FloatType(), True), 
    StructField("avg_utilization_ratio", FloatType(), True)])

#importing file from volume as a new dataframe
metrics_df = spark.read.csv(metrics_path, header=True, schema=metrics_schema)

#DEBUG: displaying dataframe
metrics_df.show()

metrics_pdf = metrics_df.toPandas() 
metrics_pdf.to_csv('metrics_table_bronze.csv')

# COMMAND ----------

# schema for customer table
customer_path = "dbfs:/Volumes/main/intern_project_2024/intern_datasets/customer_table.csv/"
customer_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("customer_age", FloatType(), True),
    StructField("gender", StringType(), True),
    StructField("dependent_count", FloatType(), True),
    StructField("education_level", StringType(), True),
    StructField("marital_status", StringType(), True),
    StructField("income", FloatType(), True)
                             ])
customer_df = spark.read.csv(customer_path, schema=customer_schema)   

customer_df.show()

customer_pdf = customer_df.toPandas() 
customer_pdf.to_csv('customer_table_bronze.csv')



# COMMAND ----------

# schema for transactions table

transactions_path = "dbfs:/Volumes/main/intern_project_2024/intern_datasets/transactions_table.csv"

transactions_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("transaction_amount", FloatType(), True),
    StructField("transaction_date", DateType(), True),
    StructField("account_id", StringType(), True)]
                          )

transactions_df = spark.read.csv(transactions_path, schema=transactions_schema)   

transactions_df.show()

transactions_pdf = transactions_df.toPandas() 
transactions_pdf.to_csv('transactions_table_bronze.csv')
