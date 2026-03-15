import os
import pickle

import implicit
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine


def train() -> None:

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    engine = create_engine(f"postgresql://{user}:{password}@postgres:5432/ecommerce")

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("ecommerce-recsys")

    print("Loading data from PostgreSQL...")

    train_df = pd.read_sql("SELECT * FROM train_data", engine)
    test_df = pd.read_sql("SELECT * FROM test_data", engine)
    train_df["user_id"], user_id_mapping = pd.factorize(train_df["visitor_id"])
    train_df["good_id"], item_id_mapping = pd.factorize(train_df["item_id"])

    test_user_id = {
        id: i for i, id in enumerate(user_id_mapping)
    }  # for mapping train user ids to test user ids
    test_item_id = {
        id: i for i, id in enumerate(item_id_mapping)
    }  # for mapping train item ids to test item ids

    test_df["user_id"] = test_df["visitor_id"].map(test_user_id)
    test_df["good_id"] = test_df["item_id"].map(test_item_id)

    test_df = test_df.dropna(subset=["user_id", "good_id"])

    train_matrix = csr_matrix(
        (train_df["score"].astype(float), (train_df["user_id"], train_df["good_id"]))
    )

    params_list = [
        {"factors": 50, "regularization": 0.01, "iterations": 20},
        {"factors": 100, "regularization": 0.05, "iterations": 30},
        {"factors": 150, "regularization": 0.1, "iterations": 50},
    ]

    for params in params_list:
        with mlflow.start_run():
            mlflow.log_params(params)

            model = implicit.als.AlternatingLeastSquares(**params)
            model.fit(train_matrix)

            precision = calculate_precision(model, train_matrix, test_df, k=10)

            print(f"Precision@10: {precision}")
            mlflow.log_metric("precision@10", precision)

            with open("model.pkl", "wb") as f:
                pickle.dump(
                    {
                        "model": model,
                        "user_id_mapping": user_id_mapping,
                        "item_id_mapping": item_id_mapping,
                    },
                    f,
                )

            mlflow.log_artifact("model.pkl")


def calculate_precision(
    model: implicit.als.AlternatingLeastSquares,
    train_matrix: csr_matrix,
    test_df: pd.DataFrame,
    k: int = 10,
) -> float:  # function for accuracy measurement
    count = 0
    test_users = test_df["user_id"].unique()[:500]  # testing with 500 test visitors

    for id in test_users:
        id = int(id)
        recommendation = model.recommend(id, train_matrix[id], N=k)[
            0
        ]  # only recomendations

        actual = test_df[test_df["user_id"] == id]["good_id"].values

        if any(np.isin(recommendation, actual)):
            count += 1

    print(f"Models approximate accuracy is {count / len(test_users)}")

    return count / len(test_users)


if __name__ == "__main__":
    train()
