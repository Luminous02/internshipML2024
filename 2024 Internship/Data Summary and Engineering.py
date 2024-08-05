# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, FloatType, DateType, IntegerType
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesRegressor


# COMMAND ----------

#importing and decompressing table with all data merged
gold_path = './Bronze, Silver, Gold/Gold/merged_table_gold.csv.gz' 
gold_pdf = pd.read_csv(gold_path, index_col=0, compression='gzip')
gold_pdf

# COMMAND ----------

#code to provide aggregate data on all numerical columns of table
#JUST USE DESCRIBE FUNCTION LOL
gold_pdf.agg(
    {
        'customer_age' :['skew'],
        'dependent_count' : ['skew'],
        'income' : ['skew'],
        'credit_limit' : ['skew'],
        'total_relationship_count' : ['skew'],
        'contacts_count_12mo' : ['skew'],
        'total_revolving_bal' : ['skew'],
        'total_amt_chng_Q4_Q1' : ['skew'],
        'total_ct_chng_Q4_Q1' : ['skew'],
        'avg_utilization_ratio' : ['skew']
    }
)

# COMMAND ----------

#One hot encoding of categorical variables
one_hot_pdf = pd.get_dummies(gold_pdf, columns = ['gender', 'education_level','marital_status','card_category', 'status'])
one_hot_pdf.sample(10)

# COMMAND ----------



# COMMAND ----------

#Feature engineering! Ideas: monthly average spending, average transactions per month

avg_monthly_spend = one_hot_pdf.avg_utilization_ratio * one_hot_pdf.credit_limit

#need drop function or else it will give already exists error
one_hot_pdf.drop('avg_monthly_spend', axis=1, inplace=True, errors='ignore')
one_hot_pdf.insert(12, 'avg_monthly_spend', avg_monthly_spend)

one_hot_pdf['transaction_date'] = pd.to_datetime(one_hot_pdf['transaction_date'])

#create month and year columns
one_hot_pdf['month'] = one_hot_pdf['transaction_date'].dt.month
one_hot_pdf['year'] = one_hot_pdf['transaction_date'].dt.year

#might cause errors here with duplicate columns when merging
transaction_counts = one_hot_pdf.groupby(['customer_id','year','month']).size().reset_index(name='num_transactions')
avg_trans_per_month = transaction_counts.groupby('customer_id')['num_transactions'].mean().reset_index()
one_hot_pdf.drop('num_transactions', axis=1, inplace=True, errors='ignore')
one_hot_pdf = pd.merge(one_hot_pdf, avg_trans_per_month, on='customer_id', how='outer')
one_hot_pdf.drop('month', axis=1, inplace=True)
one_hot_pdf.drop('year', axis=1, inplace=True)

one_hot_pdf

# COMMAND ----------

column_to_move = one_hot_pdf.pop('num_transactions')
one_hot_pdf.insert(17, 'avg_trans_per_month', column_to_move)
one_hot_pdf

# COMMAND ----------

one_hot_pdf05 = one_hot_pdf.drop_duplicates(subset=['customer_id'])

# COMMAND ----------

one_hot_pdf05

# COMMAND ----------

# fixing outliers/skew
max_income = one_hot_pdf05["income"].quantile(0.98)            
print(max_income)

max_amt_chng = one_hot_pdf05["total_amt_chng_Q4_Q1"].quantile(0.99)
print(max_amt_chng)

max_ct_chng = one_hot_pdf05["total_ct_chng_Q4_Q1"].quantile(0.99)
print(max_ct_chng)

# COMMAND ----------

#NEW TABLE WITH ALL DATA CLEANING/REMOVALS
clean_pdf = one_hot_pdf05[(one_hot_pdf05.income<max_income) & (one_hot_pdf05.total_amt_chng_Q4_Q1 < max_amt_chng) & (one_hot_pdf05.total_ct_chng_Q4_Q1 < max_ct_chng)]

# COMMAND ----------

# skew after?
one_hot_pdf.agg(
    {
        'customer_age' :['skew'],
        'dependent_count' : ['skew'],
        'income' : ['skew'],
        'credit_limit' : ['skew'],
        'total_relationship_count' : ['skew'],
        'contacts_count_12mo' : ['skew'],
        'total_revolving_bal' : ['skew'],
        'total_amt_chng_Q4_Q1' : ['skew'],
        'total_ct_chng_Q4_Q1' : ['skew'],
        'avg_utilization_ratio' : ['skew']
    }
)
#print('Rows removed:',gold_pdf.shape[0] - clean_pdf.shape[0])


# COMMAND ----------

# feature selection using VARIANCE THRESHOLD
# which removes all the low variance features from the dataset that are of no great use in modeling. (unsupervised)

# using a threshold of 0.01 would mean dropping the column where 99% of the values are similar
# removes features with low variance, meaning they are constant or nearly constant throughout the dataset
# LV features provide little information for predictive models 
selector = VarianceThreshold(threshold=0.1) #threshold can be changed!

#drop all columns that are strings
one_hot_pdf2 = one_hot_pdf05.drop(columns=['customer_id', 'account_id', 'account_open_date', 'transaction_id', 'transaction_date'])
one_hot_pdf2

#fit the data from one_hot_pdf2
selector.fit(one_hot_pdf2)

# shows the features that are high variance(True) or not(False)
print(selector.get_support())

concol = [column for column in one_hot_pdf2.columns 
          if column not in one_hot_pdf2.columns[selector.get_support()]]

final = one_hot_pdf2.drop(concol,axis=1)

mean_pdf = one_hot_pdf2.mean(axis='rows')
var_pdf = one_hot_pdf2.var(axis='rows')
summary_df = pd.DataFrame({
    'mean': mean_pdf,
    'variance': var_pdf
}).transpose()
display(summary_df)

print("\nHigh variance features:\n",final.columns)

print("\nLow variance features:")
for features in concol:
    print(features)


# To display the variance of each feature
# .var()

# COMMAND ----------

#lists all column names in one_hot_pdf2
col_headers = one_hot_pdf2.columns.tolist()

array = one_hot_pdf2.values

X = array[:,0:28]
Y = array[:,28]

# feature extraction
model = ExtraTreesRegressor(n_estimators=12)
model.fit(X, Y)

feature_importances = model.feature_importances_
print(feature_importances, col_headers[:28])

#FIRST RUN (included attrited?): [0.05338307 0.04208317 0.02864751 0.04768967 0.06612619 0.06988684 <<0.14617304>> 0.07567729 <0.19101147> 0.04007412 <<<0.09245103>>> 0.01077369 0.01088004 0.01128 0.00810629 0.01658164 0.01430057 0.0090701 0.0132878  0.00934058 0.0147881  0.01469074 0.00481175 0.00315299 0.00112639 0.00460593]

#SECOND RUN: [0.05201805 0.04202821 0.02867086 0.04938544 0.06743426 0.07163152 0.14474408 0.08040666 0.18923414 0.03660796 0.09231943 0.01113844 0.01005573 0.01122357 0.00851392 0.01653739 0.01436219 0.0088891 0.01290873 0.00935378 0.01454245 0.01422787 0.00516269 0.00296093 0.00124336 0.00439924]

#THIRD RUN (NOW INCLUDING AVG MONTHLY SPEND AND AVG MONTHLY TRANSACTIONS): [0.04597577 0.0284699  0.02050902 0.030101   0.0637918  0.06646886 <<<0.11742444>>> 0.0489881  <<0.14684352>> 0.04316612 0.02250511 0.09447194 <0.15370751> 0.00947908 0.00966512 0.00874415 0.00596669 0.01251737 0.01123932 0.00697098 0.01037281 0.00724174 0.0123886  0.01121777 0.00443716 0.00262531 0.0009244  0.00378641]

# COMMAND ----------

#testing univariate selection for feature selection
array = one_hot_pdf2.to_numpy()
X = array[:,0:28]
Y = array[:,28]

test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)

# summarize scores
univariate_pdf = pd.DataFrame({
    'category' : one_hot_pdf2.columns[0:28],
    'fit_scores' : fit.scores_
})
features = fit.transform(X)

# summarize selected features
univariate_pdf.sort_values(by=['fit_scores'],ascending=False)

# COMMAND ----------

# testing chi-squared function with categorical values in table
array = one_hot_pdf2.columns
for i in range(0,26):
    crosstab = pd.crosstab(one_hot_pdf2[array[i]], one_hot_pdf2['status_Attrited Customer'])
    print(array[i],stats.chi2_contingency(crosstab),'\n')

# COMMAND ----------

#export compressed csv file
engineered_pdf = one_hot_pdf2[['avg_monthly_spend','avg_trans_per_month','total_revolving_bal','total_ct_chng_Q4_Q1','transaction_amount','contacts_count_12mo','gender_M','gender_F','status_Attrited Customer']].copy()
engineered_pdf.to_csv('./engineered_table.csv.gz', compression='gzip')
