import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error

def evaluate_classification(x_test, y_test, run_id):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model_clf")
    preds = model.predict(x_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

    print(f"Evaluation completed | Accuracy = {acc:.3f}")

    return acc, prec, rec


def evaluate_regression(x_test, y_test, run_id):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model_reg")

    preds = model.predict(x_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)

    print(f"Evaluation completed | R2 Score = {r2:.3f}")

    return r2