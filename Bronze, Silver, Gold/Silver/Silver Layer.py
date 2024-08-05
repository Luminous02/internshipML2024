# Databricks notebook source
#import dependencies
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, FloatType, DateType, IntegerType
import pandas as pd
import plotly.express as px

# COMMAND ----------

# MAGIC %run ./configuration

# COMMAND ----------

# Do not need to run again!
# moves mounted folder into a volume
source_dir = 'dbfs:/FileStore/user/alexander.bonner@infinitive.com/credit_card_attrition/final_data'
dest_dir = 'dbfs:/Volumes/main/intern_project_2024/intern_datasets'

dbutils.fs.cp(source_dir, dest_dir, recurse=True)

# COMMAND ----------

# schema for closed accounts table
closed_path = 'dbfs:/Volumes/main/intern_project_2024/intern_datasets/closed_accounts_table.csv/'
closed_schema = StructType([StructField('customer_id', StringType(), True),
                             StructField('status', StringType(), True), 
                             StructField('account_id', StringType(), True)]
                            )

closed_df = spark.read.csv(closed_path, schema=closed_schema)
closed_df.show()   

#convert to a pandas frame for analysis and manipulation
closed_pdf = closed_df.toPandas()

#count all null values and output
closed_null_count = closed_pdf.isna().sum()
print('Number of null values:\n', closed_null_count)

closed_pdf.to_csv('closed_table_silver.csv')

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

#convert to a pandas frame for analysis and manipulation
credit_account_pdf = credit_account_df.toPandas()

#count all null values and output
account_null_count = credit_account_pdf.isna().sum()
print('Number of null values:', account_null_count)

credit_account_pdf.to_csv('credit_account_table_silver.csv')

# COMMAND ----------

#CELL: Import data to a dataframe and convert to a pandas dataframe

#file path variable
metrics_path = "dbfs:/Volumes/main/intern_project_2024/intern_datasets/credit_metrics_table.csv/"

#defining schema for credit metrics table
metrics_schema = StructType([
    StructField("customer_id", StringType(), True), 
    StructField("total_relationship_count",FloatType(), True), 
    StructField("contacts_count_12mo", FloatType(), True), 
    StructField("total_revolving_bal", FloatType(), True), 
    StructField("total_amt_chng_Q4_Q1", FloatType(), True), 
    StructField("total_ct_chng_Q4_Q1", FloatType(), True), 
    StructField("avg_utilization_ratio", FloatType(), True)])

#importing file from volume as a new dataframe
metrics_df = spark.read.csv(metrics_path, header=True, schema=metrics_schema)

#DEBUG: displaying dataframe
metrics_df.show()

#convert to a pandas frame for analysis and manipulation
metrics_pdf = metrics_df.toPandas()

#DEBUG: view pandas dataframe
#display(metrics_pdf)

#count all null values and output
metrics_null_count = metrics_pdf.isna().sum()
print('Number of null values:\n', metrics_null_count)

metrics_pdf.to_csv('metrics_table_silver.csv')

#data visualization to detect outliers

#DEBUG: total_amt_chng_Q4_Q1​ outlier visualization
#amtchng_plot = px.box(metrics_pdf, y ='total_amt_chng_Q4_Q1​')
#amtchng_plot.show()
#outliers: NONE?

# COMMAND ----------

#schema for customer table
customer_path = "dbfs:/Volumes/main/intern_project_2024/intern_datasets/customer_table.csv/"

customer_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("customer_age", FloatType(), True),
    StructField("gender", StringType(), True),
    StructField("dependent_count", FloatType(), True),
    StructField("education_level", StringType(), True),
    StructField("marital_status", StringType(), True),
    StructField("income", FloatType(), True)
    #StructField("status", StringType(), True)
                            
                             ])
customer_df = spark.read.csv(customer_path, schema=customer_schema)   

customer_df.show()

#convert to a pandas frame for analysis and manipulation
customer_pdf = customer_df.toPandas()

#count all null values and output
customer_null_count = customer_pdf.isna().sum()
print('Total null values:\n',customer_null_count)

print('Percent null:\n',(customer_null_count/len(customer_pdf)*100))

#selects and displays top n entries with null in education_level column
educ_null = customer_pdf[customer_pdf['education_level'].isna()]
print(educ_null.head(10).to_string())

#selects and displays top n entries with null in marital_status column
marital_null = customer_pdf[customer_pdf['marital_status'].isna()]
print(marital_null.head(10).to_string())

#selects and displays top n entries with null in marital_status column
income_null = customer_pdf[customer_pdf['income'].isna()]
print(income_null.head(10).to_string())

#data imputation for marital_status (mode imputation)
customer_pdf['marital_status'] = customer_pdf['marital_status'].fillna(customer_pdf['marital_status'].mode()[0])

#data imputation for education_level (mode imputation)
customer_pdf['education_level'] = customer_pdf['education_level'].fillna(customer_pdf['education_level'].mode()[0])

#data imputation for income column (median imputation)
customer_pdf['income'] = customer_pdf['income'].fillna(customer_pdf['income'].median())

print('<<AFTER IMPUTATION>>')

print(customer_pdf.head(30).to_string())

customer_pdf.to_csv('customer_table_silver.csv')

#data visualization to detect outliers

#DEBUG: age outlier visualization
#age_plot = px.box(customer_pdf, y ='customer_age')
#age_plot.show()

#DEBUG: dependents outlier visualization
#dependent_plot = px.box(customer_pdf, y ='dependent_count')
#dependent_plot.show()

#DEBUG: income outlier visualization
#income_plot = px.box(customer_pdf, y ='income')
#income_plot.show()

# COMMAND ----------

# schema for transactions table

transactions_path = "dbfs:/Volumes/main/intern_project_2024/intern_datasets/transactions_table.csv"

transactions_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("transaction_amount", FloatType(), True),
    StructField("transaction_date", DateType(), True),
    StructField("account_id", StringType(), True)]
                          )

transactions_table = spark.read.csv(transactions_path, schema=transactions_schema)   

transactions_table.show()

#convert to a pandas frame for analysis and manipulation
transactions_table_pdf = transactions_table.toPandas()

#count all null values and output
trans_null_count = transactions_table_pdf.isna().sum()
print('Number of null values:\n', trans_null_count)

transactions_table_pdf.to_csv('transactions_table_silver.csv')
