import json
from pathlib import Path


class Metrics(object):
    metrics = {}

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def log_metrics(self, value, model_name: str, metric_name: str, is_train: bool = False):
        self.metrics["{}__{}__{}".format({True: "train", False: "test"}[is_train], model_name, metric_name)] = value

    def get_metrics(self, model_name: str, metric_name: str, is_train: bool = False):
        return self.metrics["{}__{}__{}".format({True: "train", False: "test"}[is_train], model_name, metric_name)]

    def save_metrics(self, file_path: Path):
        with open(file_path, "w") as f:
            f.write(json.dumps(self.metrics, indent=4))


metrics = Metrics()
