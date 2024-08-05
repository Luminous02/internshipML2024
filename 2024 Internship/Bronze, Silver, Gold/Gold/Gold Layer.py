# Databricks notebook source
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, FloatType, DateType, IntegerType
import pandas as pd
import numpy as np

# COMMAND ----------

# Read in the closed table from silver folder
closed_path = '../Silver/closed_table_silver.csv' 
closed_pdf = pd.read_csv(closed_path, index_col=0)
closed_pdf

# COMMAND ----------

# Read in the credit account table from silver folder
credit_account_path = '../Silver/credit_account_table_silver.csv' 
credit_account_pdf = pd.read_csv(credit_account_path, index_col=0)
credit_account_pdf

# COMMAND ----------

metrics_path = '../Silver/metrics_table_silver.csv' 
metrics_pdf = pd.read_csv(metrics_path, index_col=0)
metrics_pdf

# COMMAND ----------

customer_path = '../Silver/customer_table_silver.csv' 
customer_pdf = pd.read_csv(customer_path, index_col=0)
customer_pdf

# COMMAND ----------

transactions_path = '../Silver/transactions_table_silver.csv' 
transactions_pdf = pd.read_csv(transactions_path, index_col=0)
transactions_pdf

# COMMAND ----------

merged_pdf = pd.merge(customer_pdf, closed_pdf, on='customer_id', how='outer')
merged_pdf.display()
merged_null_count = merged_pdf.isna().sum()
print('Number of null values:', merged_null_count)
print(merged_pdf.shape)

# COMMAND ----------

# Merging with Closed Accounts
merged_pdf = pd.merge(customer_pdf, closed_pdf, on='customer_id', how='outer')

#Merging with accounts
merged_pdf = pd.merge(merged_pdf, credit_account_pdf, on='customer_id', how='outer')

# Merging now with Credit Metrics
merged_pdf = pd.merge(merged_pdf, metrics_pdf, on='customer_id', how='outer')

merged_pdf.drop(labels='account_id_x', axis=1, inplace=True)
merged_pdf.rename(columns={'account_id_y':'account_id'}, inplace=True)
# Merging now with Transactions
merged_pdf = pd.merge(merged_pdf, transactions_pdf, on='account_id', how='outer')

#Drop entries that are missing data in the customer metrics table
merged_pdf.dropna(axis=0, subset='total_relationship_count', inplace=True)

# Displaying the merged 
merged_pdf.display()


#saving table GIVING ERROR: file too large

#merged_pdf.to_csv('../Gold/merged_table_gold.csv')
merged_null_count = merged_pdf.isna().sum()
print('Number of null values:', merged_null_count)
#print(merged_pdf[merged_pdf['total_relationship_count'].isna()])


# COMMAND ----------

print(merged_pdf.columns.tolist())

# COMMAND ----------

# change all null status to Existing customer
merged_pdf['status'].fillna("Existing Customer", inplace=True)
merged_null_count = merged_pdf.isna().sum()
print('Number of null values:', merged_null_count)
print(merged_pdf.shape)
merged_pdf.to_csv('../Gold/merged_table_gold.csv.gz', compression='gzip')

df = merged_pdf['status'].value_counts()
print(df)


# COMMAND ----------

#Function to save in chunks !!USE COMPRESSION INSTEAD!!
#chunk_size = 50000
#num_chunks = len(merged_pdf) // chunk_size + 1

#for i, chunk in enumerate(np.array_split(merged_pdf, num_chunks)):
#    chunk.to_csv(f"merged_chunk_{i}.csv", index=False)
