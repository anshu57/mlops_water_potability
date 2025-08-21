import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/anshu57/mlops_water_potability.mlflow")

import dagshub
dagshub.init(repo_owner='anshu57', repo_name='mlops_water_potability', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)