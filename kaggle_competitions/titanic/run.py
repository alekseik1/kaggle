from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from kaggle_competitions.metrics import Metrics, metrics

basedir = Path(__file__).parent

cat_features: list[str] = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]


def load_train_data(basedir: Path) -> pd.DataFrame:
    return pd.read_csv(basedir / "data" / "train.csv", index_col=0)


def load_test_data(basedir: Path) -> pd.DataFrame:
    return pd.read_csv(basedir / "data" / "test.csv", index_col=0)


def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df


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
    return metrics


def save_metrics(basedir: Path, metrics: Metrics):
    metrics.save_metrics(basedir / "metrics.json")


def make_prediction(df_test: pd.DataFrame, estimator: Pipeline) -> pd.DataFrame:
    df_pred = pd.DataFrame(index=df_test.index.copy(), columns=["Survived"])
    df_pred["Survived"] = estimator.predict(df_test)
    return df_pred


def run():
    df_train = load_train_data(basedir)
    df_test = load_test_data(basedir)
    df_train, df_test = data_preprocess(df_train), data_preprocess(df_test)
    y_true = df_train["Survived"]
    pipe = Pipeline([("features", features_pipeline()), ("model", model())])
    pipe.fit(X=df_train, y=y_true)
    metrics = log_metrics(data=df_train, y_true=y_true, estimator=pipe, is_train=True)
    save_metrics(basedir, metrics)
    df_pred = make_prediction(df_test, pipe)
    df_pred.to_csv(basedir / "data" / "submission.csv", index=True)


if __name__ == "__main__":
    run()
