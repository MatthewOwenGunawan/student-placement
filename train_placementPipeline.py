import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_classification_model(x_train, y_train):
    """
    Melatih model Klasifikasi (Placement Status) menggunakan Random Forest Classifier.
    """
    os.makedirs("artifacts", exist_ok=True)
    
    cat_feat = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_feat = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numeric_preprocess = Pipeline([
        ('num_imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_preprocess = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocess = ColumnTransformer(transformers=[
        ('numPreprocess', numeric_preprocess, num_feat),
        ('catPreprocess', categorical_preprocess, cat_feat)],
        remainder='drop'
    )

    placement_pred = Pipeline([
        ('preprocessing', preprocess),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=-1))
    ])
    mlflow.set_experiment("Placement Status Prediction")
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)             
        mlflow.log_param("min_samples_leaf", 5)      
        mlflow.log_param("model_type", "RandomForestClassifier")
        placement_pred.fit(x_train, y_train)
        joblib.dump(placement_pred, "artifacts/model_clf.pkl")
        mlflow.sklearn.log_model(placement_pred, artifact_path="model_clf")

    return run.info.run_id


def train_regression_model(x_train, y_train):
    os.makedirs("artifacts", exist_ok=True)
    
    valid_idx = y_train.dropna().index
    x_train_clean = x_train.loc[valid_idx]
    y_train_clean = y_train.loc[valid_idx]
    
    cat_feat = x_train_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    num_feat = x_train_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numeric_preprocess = Pipeline([
        ('num_imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_preprocess = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocess = ColumnTransformer(transformers=[
        ('numPreprocess', numeric_preprocess, num_feat),
        ('catPreprocess', categorical_preprocess, cat_feat)],
        remainder='drop'
    )

    salary_pred = Pipeline([
        ('preprocessing', preprocess),
        ('regressor', RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=-1))
    ])

    mlflow.set_experiment("Salary Package Prediction")
    
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("model_type", "RandomForestRegressor")

        salary_pred.fit(x_train_clean, y_train_clean)

        joblib.dump(salary_pred, "artifacts/model_reg.pkl")
        mlflow.sklearn.log_model(salary_pred, artifact_path="model_reg")

    return run.info.run_id