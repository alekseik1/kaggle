import json

from kaggle_competitions.metrics import Metrics


def test_metrics_singleton():
    m1 = Metrics()
    m2 = Metrics()
    assert id(m1) == id(m2)


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
