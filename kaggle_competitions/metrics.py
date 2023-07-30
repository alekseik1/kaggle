import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd


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

    def as_pandas(self):
        return pd.DataFrame.from_dict(self.metrics)

    def from_cv(self, model_name: str, cv: dict[str, np.ndarray]):
        for k, v in cv.items():
            tmp = k.split("test_")
            if len(tmp) == 1:
                continue
            metric_name = tmp[1]
            self.log_metrics(value=np.mean(v), model_name=model_name, metric_name=metric_name)

    def save_metrics(self, file_path: Path):
        with open(file_path, "w") as f:
            f.write(json.dumps(self.as_dict(), indent=4))

    def clear(self):
        self.metrics = collections.defaultdict(dict)


metrics = Metrics()
