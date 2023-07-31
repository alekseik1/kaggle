from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from kaggle_competitions.metrics import metrics

basedir = Path(__file__).parent
random_state = 42
target_col = "Transported"
scoring_funcs = ("accuracy", "precision", "recall")


def load_train(dir_: Path = basedir) -> pd.DataFrame:
    return pd.read_csv(dir_ / "data" / "train.csv", index_col=0)


def load_test(dir_: Path = basedir) -> pd.DataFrame:
    return pd.read_csv(dir_ / "data" / "test.csv", index_col=0)


def run_baseline(x: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame) -> np.ndarray:
    clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
    clf.fit(x, y)
    log_metrics(basedir, x, y, clf, model_name="baseline")
    return clf.predict(test_df)


def log_metrics(dir_: Path | None, x, y_true, estimator, model_name):
    for name, func in [
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("log_loss", log_loss),
    ]:
        metrics.log_metrics(func(y_true, estimator.predict(x)), model_name, name)
    if dir_ is not None:
        metrics.save_metrics(dir_ / "metrics.json")


def save_submit(dir_: Path, predictions: np.ndarray, test_df: pd.DataFrame):
    pd.DataFrame({"Transported": ["True" if x else "False" for x in predictions]}, index=test_df.index).to_csv(
        dir_ / "data" / "submission.csv"
    )


def run_model(x, y):
    # space = {
    #     "num_trees": hp.quniform("num_trees", 100, 200, 10),
    #     "learning_rate": hp.uniform("learning_rate", 0, 0.3),
    #     "depth": hp.quniform("depth", 2, 9, 1),
    #     "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 10),
    #     "bagging_temperature": hp.uniform("bagging_temperature", 2, ),
    #     "seed": 0,
    # }
    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "fillna",
                            ColumnTransformer(
                                [
                                    (
                                        "Age",
                                        make_pipeline(
                                            SimpleImputer(strategy="most_frequent"), QuantileTransformer(n_quantiles=50)
                                        ),
                                        ["Age"],
                                    ),
                                    (
                                        "quant_money",
                                        make_pipeline(
                                            SimpleImputer(strategy="constant", fill_value=0), QuantileTransformer()
                                        ),
                                        ["RoomService", "FoodCourt", "VRDeck", "Spa", "ShoppingMall"],
                                    ),
                                    (
                                        "false_most_freq",
                                        SimpleImputer(strategy="most_frequent", fill_value=False),
                                        ["CryoSleep", "VIP"],
                                    ),
                                    (
                                        "HomePlanet",
                                        make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()),
                                        ["HomePlanet"],
                                    ),
                                ],
                                verbose_feature_names_out=False,
                            ),
                        ),
                        (
                            "add_cab_features",
                            make_pipeline(
                                FunctionTransformer(
                                    lambda df: df["Cabin"].str.split("/", expand=True),
                                    feature_names_out=lambda self, _: ["CabLetter", "CabNumber", "CabSide"],
                                ),
                            ),
                        ),
                    ]
                ),
            ),
            (
                "post_process",
                ColumnTransformer(
                    [
                        (
                            "CabNumber",
                            make_pipeline(
                                SimpleImputer(fill_value=-1),
                                FunctionTransformer(
                                    lambda df: df.astype(int), feature_names_out=lambda _, _0: ["CabNumber"]
                                ),
                            ),
                            [-2],
                        ),
                        ("ohe", OneHotEncoder(), [-1, -3]),
                    ],
                    remainder="passthrough",
                ),
            ),
            ("model", CatBoostClassifier(random_seed=42, verbose=False)),
        ]
    )

    # def objective(space):
    #     model = pipe.set_params(
    #         model__num_trees=int(space["num_trees"]),
    #         model__learning_rate=space["learning_rate"],
    #         model__depth=int(space["depth"]),
    #         model__l2_leaf_reg=space["l2_leaf_reg"],
    #         model__bagging_temperature=space["bagging_temperature"],
    #     )
    #     cv = cross_validate(pipe, x, y, scoring=scoring_funcs, n_jobs=-1)
    #     return {"loss": 1 - np.mean(cv["test_accuracy"]), "status": STATUS_OK, "model": model}

    # trials = Trials()
    # best_hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=500, trials=trials)
    best_hyperparams = {
        "bagging_temperature": 2.380249955633156,
        "depth": 8.0,
        "l2_leaf_reg": 7.342778007475186,
        "learning_rate": 0.028814427613815457,
        "num_trees": 110.0,
    }
    pipe.set_params(**{f"model__{k}": v for k, v in best_hyperparams.items()})
    cv = cross_validate(pipe, x, y, scoring=scoring_funcs, n_jobs=-1)
    metrics.from_cv("model", cv)
    metrics.save_metrics(basedir / "metrics.json")
    pipe.fit(x, y)
    return pipe


def run():
    df_train, df_test = load_train(), load_test()
    x, y = df_train.drop(columns=[target_col]), df_train[target_col].astype(int)

    # baseline_pred = run_baseline(x, y, df_test)
    model = run_model(x, y)
    pred = model.predict(df_test)
    save_submit(basedir, pred, df_test)


if __name__ == "__main__":
    run()
