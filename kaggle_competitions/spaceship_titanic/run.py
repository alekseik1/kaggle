from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

from kaggle_competitions.metrics import metrics

basedir = Path(__file__).parent
random_state = 42
target_col = "Transported"


def load_train(dir_: Path = basedir) -> pd.DataFrame:
    return pd.read_csv(dir_ / "data" / "train.csv")


def load_test(dir_: Path = basedir) -> pd.DataFrame:
    return pd.read_csv(dir_ / "data" / "test.csv")


def baseline_model() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def log_metrics(x, y_true, estimator, model_name):
    for name, func in [
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("log_loss", log_loss),
    ]:
        metrics.log_metrics(func(y_true, estimator.predict(x)), model_name, name)
    metrics.save_metrics(basedir / "metrics.json")


def run():
    df_train, df_test = load_train(), load_test()
    x, y = df_train.drop(columns=[target_col]), df_train[target_col]

    model = baseline_model()
    model.fit(x, y)
    log_metrics(x, y, model, model_name="baseline")
    pd.DataFrame({"PassengerId": df_test["PassengerId"], "Transported": model.predict(df_test)}).to_csv(
        basedir / "data" / "submission.csv", index=False
    )


if __name__ == "__main__":
    run()
