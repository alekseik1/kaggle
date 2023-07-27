from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from kaggle_competitions.metrics import metrics

basedir = Path(__file__).parent

cat_features: list[str] = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]


def features_pipeline():
    return ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(dtype="int", sparse=False, drop="first", handle_unknown="ignore"),
                cat_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model():
    return RandomForestClassifier(random_state=42)


def log_metrics(data: pd.DataFrame, y_true: np.ndarray, estimator: Pipeline, is_train: bool):
    y_pred = estimator.predict(data)
    metrics.log_metrics(
        metric_name="precision", model_name="random_forest", value=precision_score(y_true, y_pred), is_train=is_train
    )
    metrics.log_metrics(
        metric_name="recall", model_name="random_forest", value=recall_score(y_true, y_pred), is_train=is_train
    )
    metrics.log_metrics(
        metric_name="log_loss", model_name="random_forest", value=log_loss(y_true, y_pred), is_train=is_train
    )
    metrics.save_metrics(basedir / "metrics.json")


pipe = Pipeline([("features", features_pipeline()), ("model", model())])


def run():
    df_train = pd.read_csv(basedir / "data" / "train.csv", index_col=0)
    df_test = pd.read_csv(basedir / "data" / "test.csv", index_col=0)
    y_true = df_train["Survived"]
    pipe.fit(X=df_train, y=y_true)
    log_metrics(data=df_train, y_true=y_true, estimator=pipe, is_train=True)
    df_pred = pd.DataFrame(index=df_test.index.copy(), columns=["Survived"])
    df_pred["Survived"] = pipe.predict(df_test)
    df_pred.to_csv(basedir / "data" / "submission.csv", index=True)


if __name__ == "__main__":
    run()
