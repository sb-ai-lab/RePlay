import warnings
from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

from replay.data.dataset import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.models import PopRec

pyspark = pytest.importorskip("pyspark")

cols = ["user_id", "item_id", "rating"]

data = [
    [1, 1, 0.5],
    [1, 2, 1.0],
    [2, 2, 0.1],
    [2, 3, 0.8],
    [3, 3, 0.7],
    [4, 3, 1.0],
]
feature_schema = FeatureSchema(
    [
        FeatureInfo(
            column="user_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.QUERY_ID,
        ),
        FeatureInfo(
            column="item_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
        ),
        FeatureInfo(
            column="rating",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.RATING,
        ),
    ]
)


@pytest.fixture(scope="module")
def pandas_interactions():
    return pd.DataFrame(data, columns=cols)


@pytest.fixture(scope="module")
def spark_interactions(spark, pandas_interactions):
    return spark.createDataFrame(pandas_interactions)


@pytest.fixture(scope="module")
def polars_interactions(pandas_interactions):
    return pl.DataFrame(pandas_interactions)


@pytest.fixture(scope="function")
def datasets(spark_interactions, polars_interactions, pandas_interactions):
    return {
        "pandas": Dataset(feature_schema, pandas_interactions),
        "polars": Dataset(feature_schema, polars_interactions),
        "spark": Dataset(feature_schema, spark_interactions),
    }


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model",
    [
        PopRec(),
        PopRec(use_rating=True),
    ],
    ids=["pop_rec", "pop_rec_with_rating"],
)
def test_fit_predict_the_same_framework_spark(datasets, base_model):
    results = {}
    for framework, df in datasets.items():
        model = deepcopy(base_model)
        results.update({f"{framework}": model.fit_predict(df, k=1)})

    pandas_res = results["pandas"]
    pandas_res = pandas_res.sort_values(["user_id", "item_id"])
    spark_res = results["spark"].sort("user_id", "item_id").toPandas()
    assert pandas_res.equals(spark_res), "Dataframes are not equals"


@pytest.mark.core
@pytest.mark.parametrize(
    "base_model",
    [
        PopRec(),
        PopRec(use_rating=True),
    ],
    ids=["pop_rec", "pop_rec_with_rating"],
)
def test_fit_predict_the_same_framework_polars(datasets, base_model):
    results = {}
    for framework, df in datasets.items():
        model = deepcopy(base_model)
        results.update({f"{framework}": model.fit_predict(df, k=1)})

    pandas_res = results["pandas"]
    polars_res = results["polars"].to_pandas()
    assert pandas_res.equals(polars_res)


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments, predict_framework",
    [
        (PopRec, {}, "pandas"),
        (PopRec, {}, "spark"),
        (PopRec, {}, "polars"),
        (PopRec, {"use_rating": True}, "pandas"),
        (PopRec, {"use_rating": True}, "spark"),
        (PopRec, {"use_rating": True}, "polars"),
    ],
    ids=[
        "pop_rec_pandas",
        "pop_rec_spark",
        "pop_rec_polars",
        "pop_rec_with_rating_pandas",
        "pop_rec_with_rating_spark",
        "pop_rec_with_rating_polars",
    ],
)
def test_fit_predict_different_frameworks_spark(datasets, base_model, arguments, predict_framework):
    results = {}
    model_default = base_model(**arguments)
    base_res = model_default.fit_predict(datasets["spark"], k=1).sort("user_id", "item_id").toPandas()
    for train_framework, df in datasets.items():
        if (
            predict_framework in ["polars", "pandas"] and train_framework in ["polars", "pandas"]
        ) or predict_framework == train_framework:
            continue
        model = base_model(**arguments)
        model.fit(df)
        res = None
        if predict_framework == "pandas":
            model.to_pandas()
            df.to_pandas()
            res = model.predict(df, k=1).sort_values(["user_id", "item_id"])
        elif predict_framework == "spark":
            model.to_spark()
            df.to_spark()
            res = model.predict(df, k=1).sort("user_id", "item_id").toPandas()
        if res is not None:
            results.update({f"{train_framework}_{predict_framework}": res})

    for type_of_convertation, dataframe in results.items():
        assert dataframe.equals(base_res), f"Not equal dataframes in {type_of_convertation} pair of train-predict"


def test_fit_predict_different_frameworks_pandas_polars(datasets, base_model, arguments, predict_framework):
    results = {}
    model_default = base_model(**arguments)
    base_res = model_default.fit_predict(datasets["spark"], k=1).sort("user_id", "item_id").toPandas()
    for train_framework, df in datasets.items():
        if train_framework == predict_framework or train_framework == "spark":
            continue
        model = base_model(**arguments)
        model.fit(df)
        if predict_framework == "pandas":
            model.to_pandas()
            df.to_pandas()
            res = model.predict(df, k=1).sort_values(["user_id", "item_id"])
        elif predict_framework == "polars":
            model.to_polars()
            df.to_polars()
            res = model.predict(df, k=1)
            res = res.sort("user_id", "item_id").to_pandas()
        results.update({f"{train_framework}_{predict_framework}": res})

    warnings.warn(f"{type(base_res)=} , {type(datasets['spark'])=}")
    for type_of_convertation, dataframe in results.items():
        assert dataframe.equals(base_res), f"Not equal dataframes in {type_of_convertation} pair of train-predict"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (PopRec, {"use_rating": True})],
    ids=["pop_rec", "pop_rec_with_rating"],
)
def test_fit_predict_different_frameworks_pandas_polars(datasets, base_model, arguments):
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    model = base_model(**arguments)
    model.fit(pandas_df)
    model.to_polars()
    res1 = model.predict(polars_df, k=1).sort("user_id", "item_id").to_pandas()
    model = base_model(**arguments)
    model.fit(polars_df)
    model.to_pandas()
    res2 = model.predict(pandas_df, k=1).sort_values(["user_id", "item_id"])
    assert res1.equals(res2), "Not equal dataframes in pair of train-predict"
