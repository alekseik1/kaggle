from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from kaggle_competitions.metrics import Metrics, metrics

basedir = Path(__file__).parent

cat_features: list[str] = ["Pclass", "Sex", "Embarked", "Parch"]
target_col = "Survived"


def load_train_data(basedir: Path) -> pd.DataFrame:
    return pd.read_csv(basedir / "data" / "train.csv", index_col=0)


def load_test_data(basedir: Path) -> pd.DataFrame:
    return pd.read_csv(basedir / "data" / "test.csv", index_col=0)


def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna({"Embarked": "undefined", "Age": df["Age"].mean()}).drop(
        columns=set(df.columns) - set(cat_features) - {"Age", target_col, "Fare"}
    )


def model():
    return CatBoostClassifier(random_seed=42, cat_features=cat_features, verbose=False)


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
    x, y = df_train.drop(columns=[target_col]), df_train[target_col]

    pipe = GridSearchCV(model(), param_grid={"custom_loss": ["Logloss", "CrossEntropy"]})
    pipe.fit(X=x, y=y)
    metrics = log_metrics(data=x, y_true=y, estimator=pipe, is_train=True)
    save_metrics(basedir, metrics)
    df_pred = make_prediction(df_test, pipe)
    df_pred.to_csv(basedir / "data" / "submission.csv", index=True)


if __name__ == "__main__":
    run()
