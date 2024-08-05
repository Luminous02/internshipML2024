# Databricks notebook source
# MAGIC %pip install xgboost
# MAGIC %pip install mlflow

# COMMAND ----------

# Deliverables: a handful of models (4-5) trained and tracked, best model re-trained with hyperparameter tuning, best model used for batch inference on “new” data 

# libraries to use with mlflow: 
# sklearn (RandomForestClassifier, GradientBoostingClassifier, SVC, SVM?) goes well with our data
# gradient boosting libraries (XGBoost, LightGBM, and CatBoost Models) also can be used with out data
# tensorflow, pytorch (these are more used for image classification)?

# SVM, logistic regression, XG BOOST, random tree, decision tree, KNN, Naive Bayes

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pyspark.pandas as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


# COMMAND ----------

#Splitting testing data into training, testing, and validation sets

#import table
eng_path = '../engineered_table.csv.gz'
engineered_pdf = pd.read_csv(eng_path, index_col=0, compression='gzip')

#split set into 70/15/15
train, validate, test = np.split(engineered_pdf.sample(frac=1, random_state=42),[int(0.7*len(engineered_pdf)), int(0.85*len(engineered_pdf))])

#shuffle training set
train = train.sample(frac=1).reset_index(drop=True)
train

#scale data sets
sc = StandardScaler()
scaled_train = sc.fit_transform(train.drop(columns=['status_Attrited Customer']))
scaled_test = sc.fit_transform(test.drop(columns=['status_Attrited Customer']))
scaled_validate = sc.fit_transform(validate.drop(columns=['status_Attrited Customer']))

#convert scaled sets back into pandas dataframes
scaled_train_pd = pd.DataFrame(scaled_train)
scaled_test_pd = pd.DataFrame(scaled_test)
scaled_validate_pd = pd.DataFrame(scaled_validate)

# COMMAND ----------

display(engineered_pdf)

# COMMAND ----------

engineered_pdf.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment ID : 582307333748243

# COMMAND ----------

# Logistic Regression

# imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# target/what we want to predict
target_column = 'status_Attrited Customer'

# splitting data into feature and target
X_train = train.drop(columns=[target_column])
y_train = train[target_column]

X_validate = validate.drop(columns=[target_column])
y_validate = validate[target_column]

X_test = test.drop(columns=[target_column])
y_test = test[target_column]

#
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validate_scaled = scaler.transform(X_validate)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# prediction
y_pred = model.predict(X_test_scaled)

# confusion matrix to show performance
# classification report to show performance (precision, recall, f1)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# print reports/results
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)

# NOT DONE RAHHHH
# with mlflow.start_run():
    mlflow.set_tag('mlflow.runName', 'Logistic Regression')
    
    mlflow.log_metric("")
    mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, pred))
    mlflow.log_metric("precision", metrics.precision_score(y_test, pred))
    mlflow.log_metric("recall", metrics.recall_score(y_test, pred))

    mlflow.sklearn.log_model(clf, 'regression_model')

# COMMAND ----------

# MAGIC %md
# MAGIC ###XGBOOST

# COMMAND ----------

# XG Boost
import mlflow.xgboost
import xgboost as xgb

#Thise defines a target column that is easier to call 
target_column = 'status_Attrited Customer'

#Splitting data into feature and target
X_train = train.drop(columns=[target_column])
y_train = train[target_column]

X_validate = validate.drop(columns=[target_column])
y_validate = validate[target_column]

X_test = test.drop(columns=[target_column])
y_test = test[target_column]

# Convert data to DMatrix
dtrain = xgb.DMatrix(scaled_train_pd, label=y_train)
dvalidate = xgb.DMatrix(scaled_validate_pd, label=y_validate)
dtest = xgb.DMatrix(scaled_test_pd, label=y_test) 

#sets the model's parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.1,
    'eval_metric': 'logloss'
}
    
num_boost_round = 200
evals = [(dtrain, 'train'), (dvalidate, 'validate')]

# Train the model and log with MLflow
with mlflow.start_run():
    mlflow.set_tag('mlflow.runName', 'XG_Boost')
    
    #Trains model and has early stopping rounds if data is not improving an longer, preventing overfitting
    bst = xgb.train(params, dtrain, num_boost_round, evals,)#early_stopping_rounds=10)

    #Helps make Predictions
    y_pred_prob = bst.predict(dtest)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]


    # Calculates Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    

    print('ROC AUC Score:', roc_auc)
    print("Test F1 Score:", test_f1)
    print("Test Balanced Accuracy:", test_balanced_accuracy)
    print("Test Accuracy", test_accuracy)

    
    #Logging parameters
    mlflow.log_param('objective', params['objective'])
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('eta', params['eta'])
    mlflow.log_param('eval_metric', params['eval_metric'])

    #logging Metrics
    mlflow.log_metric('test_accuracy', test_accuracy)

    mlflow.xgboost.log_model(bst, artifact_path="model")

    mlflow.sklearn.log_model(bst, 'xgboost_model')
    mlflow.end_run()


# COMMAND ----------

# MAGIC %md
# MAGIC # K-Nearest Neighbor

# COMMAND ----------

# KNN
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

#creating a knn instance (default k=5)
knn=KNeighborsClassifier(n_neighbors=265)

#training the KNN classifier 
y_train = train['status_Attrited Customer']
knn.fit(scaled_train_pd, y_train) 

#predicting the test set results
predictions= knn.predict(scaled_test_pd)

y_test = test['status_Attrited Customer']
accuracy = accuracy_score(y_test, predictions)
test_f1 = f1_score(y_test, predictions)
test_balanced_accuracy = balanced_accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print(f'ROC AUC Score: {roc_auc}')
print("Test F1 Score:", test_f1)
print("Test Balanced Accuracy:", test_balanced_accuracy)
print('Accuracy:', accuracy)


with mlflow.start_run():
    mlflow.set_tag('mlflow.runName', 'KNN')
    
    mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, predictions))
    mlflow.log_metric("precision", metrics.precision_score(y_test, predictions))
    mlflow.log_metric("recall", metrics.recall_score(y_test, predictions))

    mlflow.sklearn.log_model(knn, 'knn_model')

mlflow.end_run()


# COMMAND ----------

# MAGIC %md
# MAGIC ## SVM Support Vector Machine

# COMMAND ----------

#SVM

from sklearn import svm
from sklearn import metrics

y = train['status_Attrited Customer'].copy()
y_test = test['status_Attrited Customer'].copy()

lsvc = svm.SVC(kernel='linear')
lsvc.fit(scaled_train_pd, y)
pred = lsvc.predict(scaled_test_pd)

roc_auc = roc_auc_score(y_test, pred)

print(f'ROC AUC Score: {roc_auc}')
print("Accuracy:",metrics.accuracy_score(y_test, pred))
print("Precision:",metrics.precision_score(y_test, pred))
print("Recall:",metrics.recall_score(y_test, pred))
print("F1 Score:",metrics.f1_score(y_test, pred))

with mlflow.start_run():
    mlflow.set_tag('mlflow.runName', 'SVC')

    mlflow.log_metric("AUC ROC score", roc_auc)
    mlflow.log_metric("F1 score", f1_score(y_test, pred))
    mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
    mlflow.log_metric("precision", precision_score(y_test, pred))
    mlflow.log_metric("recall", recall_score(y_test, pred))

    mlflow.sklearn.log_model(lsvc, 'svc_model')

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest Classifier (training)

# COMMAND ----------

#set up parameter variables/values for classifier
n_estimators = 100
max_depth = 10

#create the classifier
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

#drop target column from training and test data, create target variable from test set
rf_train = train.drop(columns=['status_Attrited Customer'])
rf_test = test.drop(columns=['status_Attrited Customer'])
test_targ = train['status_Attrited Customer'].copy()

#train random forest model and predict new
rf.fit(rf_train,train['status_Attrited Customer'])

#train dummy model for comparison
dummy_cls = DummyClassifier()
dummy_cls.fit(rf_train, train['status_Attrited Customer'])

rf_predict = rf.predict(rf_train)
dummy_predict = dummy_cls.predict(rf_train)

#sample schema from table entry for mlflow logging
rf_example = rf_train.sample(1)

#log data to MLFlow
with mlflow.start_run():
    mlflow.set_tag('mlflow.runName', 'Random Forest Classifier')

    #log parameters for random forest
    mlflow.log_param('rf_n_estimators', n_estimators)
    mlflow.log_param('rf_max_depth', max_depth)

    #logging metrics for random forest
    mlflow.log_metric("rf_auc", roc_auc_score(test_targ, rf_predict))
    mlflow.log_metric("rf_f1", f1_score(test_targ, rf_predict))
    mlflow.log_metric("rf_accuracy", accuracy_score(test_targ, rf_predict))
    mlflow.log_metric("rf_precision", precision_score(test_targ, rf_predict))
    mlflow.log_metric("rf_recall", recall_score(test_targ, rf_predict))

    #logging metrics for dummy model
    mlflow.log_metric("dummy_auc", roc_auc_score(test_targ, dummy_predict))
    mlflow.log_metric("dummy_f1", f1_score(test_targ, dummy_predict))
    mlflow.log_metric("dummy_accuracy", accuracy_score(test_targ, dummy_predict))
    mlflow.log_metric("dummy_precision", precision_score(test_targ, dummy_predict))
    mlflow.log_metric("dummy_recall", recall_score(test_targ, dummy_predict))

    #log the models
    mlflow.sklearn.log_model(rf, 'random_forest_model', input_example=rf_example)
    mlflow.sklearn.log_model(dummy_cls, 'dummy_model', input_example=rf_example)
    mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ##GRID SEARCH WITH RANDOM FOREST for hyperparameter optimization 

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
import time

# defining grid values
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2)

with mlflow.start_run():
    grid_search.fit(rf_train,train['status_Attrited Customer'])

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric('best_mean_cross_val_score', grid_search.best_score_)

    test_targ = test['status_Attrited Customer'].copy()
    #test_targ_reshape = test_targ.reshape(-1, 1)

    test_score = grid_search.best_estimator_.score(rf_test, test_targ)
    mlflow.log_metric('test_score', test_score)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest w/ New Data

# COMMAND ----------

#create the new classifier
rf_final = grid_search.best_estimator_

#drop target column from validation data, create target variable from validation set
rf_validate = validate.drop(columns=['status_Attrited Customer'])
rf_targ = validate['status_Attrited Customer'].copy()

#predict on validation set
rff_predict = rf_final.predict(rf_validate)

with mlflow.start_run():
    mlflow.set_tag('mlflow.runName', 'RFC Final')

    #log parameters for final random forest
    mlflow.log_param('rff_n_estimators', 100)
    mlflow.log_param('rff_max_depth', 20)
    mlflow.log_param('rff_min_samples_leaf', 4)
    mlflow.log_param('rff_min_samples_split', 2)

    #logging metrics for final random forest
    mlflow.log_metric("rff_auc", roc_auc_score(rf_targ, rff_predict))
    mlflow.log_metric("rff_f1", f1_score(rf_targ, rff_predict))
    mlflow.log_metric("rff_accuracy", accuracy_score(rf_targ, rff_predict))
    mlflow.log_metric("rff_precision", precision_score(rf_targ, rff_predict))

    mlflow.log_metric("rff_recall", recall_score(rf_targ, rff_predict))

    rff_predict = rf_final.predict(rf_validate)


# COMMAND ----------

best_model = grid_search.best_estimator_
print(best_model)

final_score = best_model.score(rf_validate, rf_targ)
print(final_score)

with mlflow.start_run():
    mlflow.log_metrics({'final_score': final_score})

# COMMAND ----------

feature_columns = ['avg_monthly_spend', 'avg_trans_per_month', 'total_revolving_bal',
       'total_ct_chng_Q4_Q1', 'transaction_amount', 'contacts_count_12mo',
       'gender_M', 'gender_F']
feature_importances = best_model.feature_importances
features = pd.DataFrame(list(zip(feature_columns, feature_importances)), columns=["Feature", "Importance"])
features = features.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(features["Feature"], features["Importance"], color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Naive Bayes Model

# COMMAND ----------

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_train = train.drop(columns=['status_Attrited Customer'])
nb_test = test.drop(columns=['status_Attrited Customer']) 

nb_model = GaussianNB()
y_train = train.iloc[:, 8]
nb_model.fit(nb_train, y_train)
predictions = nb_model.predict(nb_test)

y_test = test.iloc[:, 8]
accuracy = accuracy_score(y_test, predictions)
test_f1 = f1_score(y_test, predictions)
test_balanced_accuracy = balanced_accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print("Accuracy: ", accuracy)
print("Test F1 Score:", test_f1)
print("Test Balanced Accuracy:", test_balanced_accuracy)
print('ROC AUC Score:', roc_auc)


with mlflow.start_run():
    mlflow.set_tag('mlflow.runName', 'NB')
    
    mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, predictions))
    mlflow.log_metric("precision", metrics.precision_score(y_test, predictions))
    mlflow.log_metric("recall", metrics.recall_score(y_test, predictions))

    mlflow.sklearn.log_model(nb_model, 'nb_model')

mlflow.end_run()

# COMMAND ----------

# Naive Bayes
#from sklearn.naive_bayes import MultinomialNB

#nb_train = train.drop(columns=['status_Attrited Customer'])
#nb_test = test.drop(columns=['status_Attrited Customer']) 

#nb_model = MultinomialNB()
#y_train = train.iloc[:, 8]
#nb_model.fit(nb_train, y_train)
#predictions = nb_model.predict(nb_test)

#y_test = test.iloc[:, 8]
#accuracy = accuracy_score(y_test, predictions)
#test_f1 = f1_score(y_test, predictions)
#test_balanced_accuracy = balanced_accuracy_score(y_test, predictions)
#roc_auc = roc_auc_score(y_test, predictions)

#print("Accuracy: ", accuracy)
#print("Test F1 Score:", test_f1)
#print("Test Balanced Accuracy:", test_balanced_accuracy)
#print('ROC AUC Score:', roc_auc)


# with mlflow.start_run():
    # mlflow.set_tag('mlflow.runName', 'NB')
    
    # mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, predictions))
    # mlflow.log_metric("precision", metrics.precision_score(y_test, predictions))
    # mlflow.log_metric("recall", metrics.recall_score(y_test, predictions))

    # mlflow.sklearn.log_model(nb_model, 'nb_model')

# mlflow.end_run()
