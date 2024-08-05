# Databricks notebook source
# MAGIC %pip install xgboost
# MAGIC %pip install mlflow

# COMMAND ----------

import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from pyspark.dbutils import DBUtils
dbwidgets = DBUtils(spark).widgets
import plotly.express as px
import plotly.graph_objects as go

# COMMAND ----------

#import table
eng_path = './engineered_table.csv.gz'
engineered_pdf = pd.read_csv(eng_path, index_col=0, compression='gzip')

# COMMAND ----------

data = {'variable':['customer_age', 'dependent_count', 'income', 'credit_limit', 'total_relationship_count', 'contacts_count_12mo', 'total_revolving_bal', 'total_amt_chng_Q4_Q1', 'total_ct_chng_Q4_Q1', 'avg_monthly_spend', 'avg_utilization_ratio', 'transaction_amount', 'avg_trans_per_month', 'gender_F', 'gender_M', 'education_level_College', 'education_level_Doctorate', 'education_level_Graduate', 'education_level_High School', 'education_level_Post-Graduate', 'education_level_Uneducated', 'marital_status_Divorced', 'marital_status_Married', 'marital_status_Single', 'card_category_Blue', 'card_category_Gold', 'card_category_Platinum', 'card_category_Silver'],'score':[0.04597577, 0.0284699, 0.02050902, 0.030101, 0.0637918, 0.06646886, 0.11742444, 0.0489881, 0.14684352, 0.04316612, 0.02250511, 0.09447194, 0.15370751, 0.00947908, 0.00966512, 0.00874415, 0.00596669, 0.01251737, 0.01123932, 0.00697098, 0.01037281, 0.00724174, 0.0123886, 0.01121777, 0.00443716, 0.00262531, 0.0009244,  0.00378641]}
rf_scores = pd.DataFrame(data=data)
rf_scores.sort_values(by='score',ascending=False,inplace=True)
rf_scores.drop(rf_scores.tail(8).index,inplace=True)
display(rf_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ![The Infinitive Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMJc8F0GoduerZeXlp9V7Dl0NdEbz_--eq9w&s)

# COMMAND ----------

# Ensure engineered_pdf is correctly defined as a string or obtained through a method call
engineered_pdf_path = "/Workspace/Users/darcy.mcfarlane@infinitive.com/2024 Internship/engineered_table.csv.gz"  # Example path, replace with your actual path or method call

# Read the CSV file into a DataFrame
epdf1 = pd.read_csv(engineered_pdf_path)

# Create dropdown widgets
dbwidgets.removeAll()  # Assuming dbwidgets is correctly initialized and removeAll is a valid method

# Populate dropdown widgets with column names from the DataFrame
column_names = [col for col in epdf1.columns]
dbwidgets.dropdown("x_axis", "status_Attrited Customer", column_names)
dbwidgets.dropdown("y_axis", "avg_trans_per_month", column_names)
# Get selected values from widgets
x_axis = dbwidgets.get("x_axis")
y_axis = dbwidgets.get("y_axis")

# Create the plot
fig = px.scatter(epdf1, x=x_axis, y=y_axis, title=f'{x_axis} VS {y_axis}')

# Display the plot
fig.show()

# COMMAND ----------

display(engineered_pdf)


# COMMAND ----------

rfc_mlflow = mlflow.search_runs(experiment_ids=["4012278545524581"])
rfc_mlflow = rfc_mlflow[rfc_mlflow.run_id == '2afa18865bb94f8fac3f01831bb18822']
rfc_mlflow.head(10)

# COMMAND ----------

#import table
gold_path = './Bronze, Silver, Gold/Gold/merged_table_gold.csv.gz'
gold_pdf = pd.read_csv(gold_path, index_col=0, compression='gzip')

# COMMAND ----------

# display gold table
display(gold_pdf)

# COMMAND ----------

gold_pdf_mean = gold_pdf.agg(
    {
        'customer_age' :['mean'],
        'dependent_count' : ['mean'],
        'income' : ['mean'],
        'contacts_count_12mo' : ['mean'],
        'total_revolving_bal' : ['mean'],
    }
)

# COMMAND ----------

display(gold_pdf_mean)

# COMMAND ----------

avg_credit_limit = gold_pdf['credit_limit'].mean()

credit_limit_df = pd.DataFrame([avg_credit_limit], columns=['avg_credit_limit'])
display(credit_limit_df)


# COMMAND ----------

total_customers = gold_pdf['customer_id'].nunique()

total_customers_df = pd.DataFrame([total_customers], columns=['total_customers'])
display(total_customers_df)

# COMMAND ----------

avg_total_revolving_bal = gold_pdf['total_revolving_bal'].mean()

avg_total_revolving_bal_df = pd.DataFrame([avg_total_revolving_bal], columns=['total_revolving_bal'])
display(avg_total_revolving_bal_df)


# COMMAND ----------

number_of_customers = engineered_pdf.filter(['customer_id','status_Attrited Customer'], axis=1)
status_count = number_of_customers.apply(pd.value_counts)
status_count.insert(0,'status',['Active','Attrited'])
display(status_count)

# COMMAND ----------

total_transactions =  gold_pdf['transaction_date'].count()

total_transactions_df = pd.DataFrame([total_transactions], columns=['transaction_date'])
display(total_transactions_df)

# COMMAND ----------

# Notes on the Dashboard:

# Add total transactions counter ( ~ 4.5 million)

# Maybe add a open vs closed counter under "Total Number of Customers" to show further breakdown

# Take out the Engineered Input Data, instead show averages of the engineered features

# Feature importance metrics (bar chart), how much each feature contributes to the model

# 


# COMMAND ----------


