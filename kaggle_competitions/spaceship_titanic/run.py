from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

from kaggle_competitions.metrics import metrics

basedir = Path(__file__).parent
random_state = 42
target_col = "Transported"


def load_train(dir_: Path = basedir) -> pd.DataFrame:
    return pd.read_csv(dir_ / "data" / "train.csv", index_col=0)


def load_test(dir_: Path = basedir) -> pd.DataFrame:
    return pd.read_csv(dir_ / "data" / "test.csv", index_col=0)


def run_baseline(x: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame) -> np.ndarray:
    clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
    clf.fit(x, y)
    log_metrics(x, y, clf, model_name="baseline")
    return clf.predict(test_df)


def log_metrics(dir_: Path, x, y_true, estimator, model_name):
    for name, func in [
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("log_loss", log_loss),
    ]:
        metrics.log_metrics(func(y_true, estimator.predict(x)), model_name, name)
    metrics.save_metrics(dir_ / "metrics.json")


def save_submit(dir_: Path, predictions: np.ndarray, test_df: pd.DataFrame):
    pd.DataFrame({"Transported": predictions}, index=test_df.index).to_csv(dir_ / "data" / "submission.csv")


def run():
    df_train, df_test = load_train(), load_test()
    x, y = df_train.drop(columns=[target_col]), df_train[target_col]

    baseline_pred = run_baseline(x, y, df_test)
    save_submit(basedir, baseline_pred, df_test)


if __name__ == "__main__":
    run()
