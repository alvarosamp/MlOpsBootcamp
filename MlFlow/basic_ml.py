import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd


def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e

def eval_function(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    return rmse, r2, mae

def main(alpha, l1_ratio):
    df = load_data()
    TARGET = "quality"
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    mlflow.set_experiment("ML-Model-1")
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, r2, mae = eval_function(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-p1", type=float, default=0.2)
    args.add_argument("--l1_ratio","-p2", type=float, default=0.3)
    parsed_args = args.parse_args()
    main(parsed_args.alpha, parsed_args.l1_ratio)

