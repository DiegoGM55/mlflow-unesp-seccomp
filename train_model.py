# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select avg(radiant_win) from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

# DBTITLE 1,Imports
# Import das libs
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import mlflow

# Import dos dados
sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")
df = sdf.toPandas()

# COMMAND ----------

# DBTITLE 1,Definição das variáveis
target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list( set(df.columns.tolist()) - set([target_column, id_column]))

y = df[target_column]
X = df[features_columns]
X

# COMMAND ----------

# DBTITLE 1,Split test e train
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

print("Número de linhas em X_train:", X_train.shape[0]) 
print("Número de linhas em X_test:", X_test.shape[0])
print("Número de linhas em y_train:", y_train.shape[0]) 
print("Número de linhas em y_test:", y_test.shape[0])

# COMMAND ----------

# DBTITLE 1,Setup do experimento MLflow
mlflow.set_experiment("/Users/diego.mota@unesp.br/dota-unesp-diego")

# COMMAND ----------

# DBTITLE 1,Run do experimento
with mlflow.start_run():
    
    mlflow.sklearn.autolog()
    
    # model = tree.DecisionTreeClassifier()
    model = LogisticRegression(estimators = 100)

    model.fit( X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print("Acuracia em treino:", acc_train)
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print("Acuracia em teste:", acc_test)
    
    mlflow.log_dict({"features": X_train.columns.tolist()})
