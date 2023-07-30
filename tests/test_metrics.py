import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_competitions.metrics import Metrics


def test_metrics_singleton():
    m1 = Metrics()
    m2 = Metrics()
    assert id(m1) == id(m2)


@pytest.fixture(autouse=True)
def clear_metrics():
    Metrics().clear()


def test_save_and_load():
    m = Metrics()
    m.log_metrics(value=1, model_name="test_model", metric_name="loss")
    assert m.get_metrics(model_name="test_model", metric_name="loss") == 1


def test_dump(tmp_path):
    m = Metrics()
    m.log_metrics(value=1, model_name="test_model", metric_name="loss")
    m.save_metrics(tmp_path / "metrics.json")
    with open(tmp_path / "metrics.json", "r") as f:
        loaded = json.load(f)
    assert loaded["loss__test_model"] == 1


def test_as_pandas():
    m = Metrics()
    m.log_metrics(value=2.1, model_name="model_1", metric_name="log_loss")
    m.log_metrics(value=3, model_name="model_1", metric_name="accuracy")
    m.log_metrics(value=1, model_name="model_2", metric_name="log_loss")
    m.log_metrics(value=5, model_name="model_2", metric_name="accuracy")
    df = m.as_pandas()
    expected_df = pd.DataFrame(index=["model_1", "model_2"], data=[[2.1, 3], [1, 5]], columns=["log_loss", "accuracy"])
    assert_frame_equal(df, expected_df)


@pytest.mark.parametrize("data", [np.array([1, 2, 4, 5]), np.array([1, 2, 3, 4])])
def test_from_cv(data):
    m = Metrics()
    m.from_cv("my_model", {"test_accuracy": data})
    assert m.get_metrics("my_model", "accuracy") == np.mean(data)
