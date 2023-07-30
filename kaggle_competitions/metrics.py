import collections
import json
from pathlib import Path


class Metrics(object):
    metrics = collections.defaultdict(dict)

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def log_metrics(self, value, model_name: str, metric_name: str):
        self.metrics[metric_name][model_name] = value

    def get_metrics(self, model_name: str, metric_name: str):
        return self.metrics[metric_name][model_name]

    def as_dict(self):
        rv = {}
        for metric_name, model_to_value in self.metrics.items():
            for model, value in model_to_value.items():
                rv[f"{metric_name}__{model}"] = value
        return rv

    def save_metrics(self, file_path: Path):
        with open(file_path, "w") as f:
            f.write(json.dumps(self.as_dict(), indent=4))


metrics = Metrics()
