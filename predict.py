# Databricks notebook source
import mlflow

model = mlflow.sklearn.load_model("models:/dota-unesp/production")

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match_new")
df = sdf.toPandas()


# COMMAND ----------

features = set(df.columns.tolist()) - set(["match_id", "radiant_win"])
X = df[features]

# COMMAND ----------

score = model.predict_proba(X)
df["proba_radian_win"] = score[:,1]

df[["match_id", "radiant_win", "proba_radian_win"]]
# 0.523766
