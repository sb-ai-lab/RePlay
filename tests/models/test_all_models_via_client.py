from os.path import dirname, join

import numpy as np
import pandas as pd
import polars as pl
import pytest

import replay
from replay.data.dataset import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.models import UCB, ClusterRec, ItemKNN, PopRec, RandomRec, Wilson, client_model_list
from replay.models.base_rec_client import NotFittedModelError
from tests.utils import (
    SparkDataFrame,
    get_dataset_any_type,
    isDataFrameEqual,
)

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf

SEED = 123

cols = ["user_id", "item_id", "rating"]

data = [
    [1, 1, 0.5],
    [1, 2, 1.0],
    [2, 2, 0.1],
    [2, 3, 0.8],
    [3, 3, 0.7],
    [4, 3, 1.0],
]

new_data = [[5, 4, 1.0]]
feature_schema_small_df = FeatureSchema(
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
        FeatureInfo(
            column="timestamp",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.TIMESTAMP,
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
        "pandas": Dataset(feature_schema_small_df, pandas_interactions),
        "polars": Dataset(feature_schema_small_df, polars_interactions),
        "spark": Dataset(feature_schema_small_df, spark_interactions),
    }


@pytest.fixture(scope="function")
def pandas_big_df():
    folder = dirname(replay.__file__)
    res = pd.read_csv(
        join(folder, "../examples/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    ).head(1000)
    return res


@pytest.fixture(scope="function")
def polars_big_df(pandas_big_df):
    return pl.from_pandas(pandas_big_df)


@pytest.fixture(scope="function")
def spark_big_df(spark, pandas_big_df):
    return spark.createDataFrame(pandas_big_df)


@pytest.fixture(scope="function")
def big_datasets(pandas_big_df, polars_big_df, spark_big_df):
    return {
        "pandas": Dataset(feature_schema, pandas_big_df),
        "polars": Dataset(feature_schema, polars_big_df),
        "spark": Dataset(feature_schema, spark_big_df),
    }


"""
@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (ItemKNN, {})],
    ids=["pop_rec", "item_knn"],
)
def test_predict_pairs_k_all_models(base_model, arguments, request):
    dataset_type = "pandas"
    model = base_model(**arguments)
    log = request.getfixturevalue("log")
    ds = get_dataset_any_type(log)
    train_dataset = get_dataset_any_type(log)
    pairs = ds.interactions.select("user_idx", "item_idx")
    model.fit(train_dataset)
    pairs_pred_k_spark = model.predict_pairs(pairs=pairs, dataset=train_dataset, k=1)
    pairs_pred_spark = model.predict_pairs(pairs=pairs, dataset=train_dataset, k=None)
    assert pairs_pred_k_spark.groupBy("user_idx").count().filter(sf.col("count") > 1).count() == 0
    assert pairs_pred_spark.groupBy("user_idx").count().filter(sf.col("count") > 1).count() > 0

    dataset_type = "pandas"
    model = base_model(**arguments)
    log = request.getfixturevalue("log_" + dataset_type)
    ds = get_dataset_any_type(log)
    train_dataset = get_dataset_any_type(log)
    pairs = ds.interactions[["user_idx", "item_idx"]]
    if model.__class__ not in client_model_list:
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
            model.fit(train_dataset)
    else:
        model.fit(train_dataset)
        pairs_pred_k = model.predict_pairs(pairs, train_dataset, k=1)
        pairs_pred = model.predict_pairs(pairs, train_dataset, k=None)
        counts_k_pd = pairs_pred_k.groupby("user_idx").size()
        counts_pd = pairs_pred.groupby("user_idx").size()
        assert all(counts_k_pd <= 1)
        assert any(counts_pd > 1)

    dataset_type = "polars"
    model = base_model(**arguments)
    log = request.getfixturevalue("log_" + dataset_type)
    ds = get_dataset_any_type(log)
    train_dataset = get_dataset_any_type(log)
    pairs = ds.interactions.select("user_idx", "item_idx")
    if model.__class__ not in client_model_list:
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
            model.fit(train_dataset)
    else:
        model.fit(train_dataset)
        pairs_pred_k = model.predict_pairs(pairs, train_dataset, k=1)
        pairs_pred = model.predict_pairs(pairs, train_dataset, k=None)
        counts_k_pl = pairs_pred_k.group_by("user_idx").count()
        counts_pl = pairs_pred.group_by("user_idx").count()
        assert all(counts_k_pl["count"].to_pandas().to_numpy() <= 1)
        assert any(counts_pl["count"].to_pandas().to_numpy() > 1)

    if model.__class__ in client_model_list:
        assert isDataFrameEqual(counts_k_pd, pairs_pred_k_spark), "Pandas predictions not equals Spark predictions"
        assert isDataFrameEqual(counts_k_pd, counts_k_pl), "Pandas predictions not equals Polars predictions"
        assert isDataFrameEqual(counts_pd, pairs_pred_spark), "Pandas predictions not equals Spark predictions"
        assert isDataFrameEqual(counts_pd, counts_pl), "Pandas predictions not equals Polars predictions"
"""


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(),
        PopRec(),
    ],
    ids=["knn", "pop_rec"],
)
def test_predict_empty_log_spark(model, request):
    log = request.getfixturevalue("log")
    empty_df = log.limit(0)
    dataset = get_dataset_any_type(log)
    pred_dataset = get_dataset_any_type(empty_df)
    model.fit(dataset)
    model.predict(pred_dataset, 1)
    model._clear_cache()


@pytest.mark.core
@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(),
        PopRec(),
    ],
    ids=["knn", "pop_rec"],
)
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"], ids=["pandas", "polars"])
def test_predict_empty_log_core(model, dataset_type, request):
    if dataset_type == "pandas":
        log = request.getfixturevalue("log_" + dataset_type)
        empty_df = log.head(0)
    elif dataset_type == "polars":
        log = request.getfixturevalue("log_" + dataset_type)
        empty_df = log.limit(0)
    else:
        msg = "Incorrect test"
        raise ValueError(msg)
    dataset = get_dataset_any_type(log)
    pred_dataset = get_dataset_any_type(empty_df)
    if model.__class__ not in client_model_list:
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
            model.fit(dataset)
            model.predict(pred_dataset, 1)
    else:
        model.fit(dataset)
        model.predict(pred_dataset, 1)


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(),
        PopRec(),
    ],
    ids=["knn", "pop_rec"],
)
def test_predict_empty_dataset_spark(model, request):
    log = request.getfixturevalue("log")
    dataset = get_dataset_any_type(log)
    pred_dataset = None
    if model.__class__ == ItemKNN:
        with pytest.raises(ValueError, match="interactions is not provided"):
            model.fit(dataset)
            model.predict(pred_dataset, 1)
    else:
        model.fit(dataset)
        model.predict(pred_dataset, 1)


@pytest.mark.core
@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(),
        PopRec(),
    ],
    ids=["knn", "pop_rec"],
)
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"], ids=["pandas", "polars"])
def test_predict_empty_dataset_core(model, dataset_type, request):
    if dataset_type == "pandas":
        log = request.getfixturevalue("log_" + dataset_type)
    elif dataset_type == "polars":
        log = request.getfixturevalue("log_" + dataset_type)
    else:
        msg = "Incorrect test"
        raise ValueError(msg)
    dataset = get_dataset_any_type(log)
    pred_dataset = None
    if model.__class__ == ItemKNN:
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
            model.fit(dataset)
    else:
        model.fit(dataset)
        model.predict(pred_dataset, 1)


@pytest.mark.spark
def test_filter_seen(dataset_type, request):
    log = request.getfixturevalue("log")
    model = PopRec()
    train_dataset = get_dataset_any_type(log.filter(sf.col("user_idx") != 0))
    pred_dataset = get_dataset_any_type(log)
    model.fit(train_dataset)
    pred = model.predict(dataset=pred_dataset, queries=[3], k=5)
    assert pred.count() == 2
    pred = model.predict(dataset=pred_dataset, queries=[0], k=5)
    assert pred.count() == 1
    pred = model.predict(dataset=pred_dataset, queries=[0], k=5, filter_seen_items=False)
    assert pred.count() == 4


@pytest.mark.core
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"], ids=["pandas", "polars"])
def test_filter_seen(dataset_type, request):
    if dataset_type == "pandas":
        log = request.getfixturevalue("log_" + dataset_type)
        model = PopRec()
        train_dataset = get_dataset_any_type(log[log["user_idx"] != 0])
        pred_dataset = get_dataset_any_type(log)
        model.fit(train_dataset)
        pred = model.predict(dataset=pred_dataset, queries=[3], k=5)
        assert len(pred) == 2
        pred = model.predict(dataset=pred_dataset, queries=[0], k=5)
        assert len(pred) == 1
        pred = model.predict(dataset=pred_dataset, queries=[0], k=5, filter_seen_items=False)
        assert len(pred) == 4
    elif dataset_type == "polars":
        log = request.getfixturevalue("log_" + dataset_type)
        model = PopRec()
        train_dataset = get_dataset_any_type(log.filter(pl.col("user_idx") != 0))
        pred_dataset = get_dataset_any_type(log)
        model.fit(train_dataset)
        pred = model.predict(dataset=pred_dataset, queries=[3], k=5)
        assert pred.height == 2
        pred = model.predict(dataset=pred_dataset, queries=[0], k=5)
        assert pred.height == 1
        pred = model.predict(dataset=pred_dataset, queries=[0], k=5, filter_seen_items=False)
        assert pred.height == 4


def fit_predict_selected(model, train_log, inf_log, queries, user_features=None, item_features=None):
    train_dataset = get_dataset_any_type(train_log, user_features=user_features, item_features=item_features)
    pred_dataset = get_dataset_any_type(inf_log, user_features=user_features, item_features=item_features)
    if not isinstance(train_log, SparkDataFrame) and model.__class__ not in client_model_list:
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
            model.fit(train_dataset)
            res = model.predict(dataset=pred_dataset, queries=queries, k=1)
            return res
    else:
        model.fit(train_dataset)
        return model.predict(dataset=pred_dataset, queries=queries, k=1)


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [ItemKNN(), PopRec()],
    ids=["knn", "pop_rec"],
)
def test_predict_new_queries_spark(model, request):
    long_log_with_features = request.getfixturevalue("long_log_with_features")
    user_features = request.getfixturevalue("user_features")
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=user_features.drop("gender"),
        queries=[0],
    )
    if pred is not None:
        assert pred.count() == 1
        assert pred.collect()[0][0] == 0
    else:
        assert model.__class__ not in client_model_list


@pytest.mark.core
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"], ids=["pandas", "polars"])
@pytest.mark.parametrize(
    "model",
    [ItemKNN(), PopRec()],
    ids=["knn", "pop_rec"],
)
def test_predict_new_queries_core(dataset_type, model, request):
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        user_features = request.getfixturevalue("user_features_" + dataset_type)
        pred = fit_predict_selected(
            model,
            train_log=long_log_with_features[long_log_with_features["user_idx"] != 0],
            inf_log=long_log_with_features,
            user_features=user_features.drop("gender", axis=1),
            queries=[0],
        )
        if pred is not None:
            assert pred is not None and pred.shape[0] == 1
            assert pred.iloc[0, 0] == 0
        else:
            assert model.__class__ not in client_model_list
    elif dataset_type == "polars":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        user_features = request.getfixturevalue("user_features_" + dataset_type)
        pred = fit_predict_selected(
            model,
            train_log=long_log_with_features.filter(pl.col("user_idx") != 0),
            inf_log=long_log_with_features,
            user_features=user_features.drop("gender"),
            queries=[0],
        )
        if pred is not None:
            assert pred.height == 1
            assert pred.to_pandas().iloc[0, 0] == 0
        else:
            assert model.__class__ not in client_model_list


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [ClusterRec(num_clusters=2), PopRec()],
    ids=["knn", "pop_rec"],
)
def test_predict_cold_spark(model, request):
    long_log_with_features = request.getfixturevalue("long_log_with_features")
    user_features = request.getfixturevalue("user_features")
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        user_features=user_features.drop("gender"),
        queries=[0],
    )
    if pred is not None:
        assert pred.count() == 1
        assert pred.collect()[0][0] == 0
    else:
        assert model.__class__ not in client_model_list


@pytest.mark.core
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"], ids=["pandas", "polars"])
@pytest.mark.parametrize(
    "model",
    [ClusterRec(num_clusters=2), PopRec()],
    ids=["knn", "pop_rec"],
)
def test_predict_cold_core(dataset_type, model, request):
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        user_features = request.getfixturevalue("user_features_" + dataset_type)
        pred = fit_predict_selected(
            model,
            train_log=long_log_with_features[long_log_with_features["user_idx"] != 0],
            inf_log=long_log_with_features[long_log_with_features["user_idx"] != 0],
            user_features=user_features.drop("gender", axis=1),
            queries=[0],
        )
        if pred is not None:
            assert pred is not None and pred.shape[0] == 1
            assert pred.iloc[0, 0] == 0
        else:
            assert model.__class__ not in client_model_list
    elif dataset_type == "polars":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        user_features = request.getfixturevalue("user_features_" + dataset_type)
        pred = fit_predict_selected(
            model,
            train_log=long_log_with_features.filter(pl.col("user_idx") != 0),
            inf_log=long_log_with_features.filter(pl.col("user_idx") != 0),
            user_features=user_features.drop("gender"),
            queries=[0],
        )
        if pred is not None:
            assert pred.height == 1
            assert pred.to_pandas().iloc[0, 0] == 0
        else:
            assert model.__class__ not in client_model_list


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (ItemKNN, {})],
    ids=["pop_rec", "item_knn"],
)
def test_fit_predict_item_features(base_model, arguments, request):
    dataset_type = "spark"
    model = base_model(**arguments)
    long_log_with_features = request.getfixturevalue("long_log_with_features")
    item_features = request.getfixturevalue("item_features")
    pred_spark = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        item_features=item_features.drop("color", "class"),
        queries=[0],
    )
    if pred_spark is not None:
        assert pred_spark.count() == 1
        assert pred_spark.collect()[0][0] == 0
    else:
        assert model.__class__ not in client_model_list
    dataset_type = "pandas"
    model = base_model(**arguments)
    long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
    item_features = request.getfixturevalue("item_features_" + dataset_type)
    pred_pd = fit_predict_selected(
        model,
        train_log=long_log_with_features[long_log_with_features["user_idx"] != 0],
        inf_log=long_log_with_features,
        item_features=item_features.drop(["color", "class"], axis=1),
        queries=[0],
    )
    if pred_pd is not None:
        assert pred_pd.shape[0] == 1
        assert pred_pd.iloc[0, 0] == 0
    else:
        assert model.__class__ not in client_model_list
    dataset_type = "polars"
    model = base_model(**arguments)
    long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
    item_features = request.getfixturevalue("item_features_" + dataset_type)
    pred_pl = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(pl.col("user_idx") != 0),
        inf_log=long_log_with_features,
        item_features=item_features.drop("class", "color"),
        queries=[0],
    )
    if pred_spark is not None and pred_pl is not None:
        assert isDataFrameEqual(pred_spark, pred_pl), "Spark predictions not equals Polars predictions"
    if pred_pl is not None and pred_pd is not None:
        assert isDataFrameEqual(pred_pl, pred_pd), "Polars predictions not equals Pandas predictions"


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [
        PopRec(),
        ItemKNN(),
    ],
    ids=["pop_rec", "knn"],
)
def test_predict_pairs_to_file_spark(model, tmp_path, request):
    long_log_with_features = request.getfixturevalue("long_log_with_features")
    train_dataset = get_dataset_any_type(long_log_with_features)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit(train_dataset)
    model.predict_pairs(
        dataset=train_dataset,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        recs_file_path=path,
    )
    pred_cached = model.predict_pairs(
        dataset=train_dataset,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        recs_file_path=None,
    )
    pred_from_file = pd.read_parquet(path)
    assert isDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [
        PopRec(),
        ItemKNN(),
    ],
    ids=["pop_rec", "knn"],
)
def test_predict_pairs_to_file_spark(model, tmp_path, request):
    long_log_with_features = request.getfixturevalue("long_log_with_features")
    train_dataset = get_dataset_any_type(long_log_with_features)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit(train_dataset)
    model.predict_pairs(
        dataset=train_dataset,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        recs_file_path=path,
    )
    pred_cached = model.predict_pairs(
        dataset=train_dataset,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        recs_file_path=None,
    )
    pred_from_file = pd.read_parquet(path)
    assert isDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.core
@pytest.mark.parametrize(
    "model",
    [
        PopRec(),
        ItemKNN(),
    ],
    ids=["pop_rec", "knn"],
)
@pytest.mark.parametrize("dataset_type", ["spark", "pandas", "polars"], ids=["spark", "pandas", "polars"])
def test_predict_pairs_to_file_core(model, dataset_type, tmp_path, request):
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        train_dataset = get_dataset_any_type(long_log_with_features)
        path = str((tmp_path / "pred.parquet").resolve().absolute())
        if model.__class__ not in client_model_list:
            with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
                model.fit(train_dataset)
                model.predict_pairs(
                    dataset=train_dataset,
                    pairs=long_log_with_features[long_log_with_features["user_idx"] == 1][["user_idx", "item_idx"]],
                    recs_file_path=path,
                )
                pred_cached = model.predict_pairs(
                    dataset=train_dataset,
                    pairs=long_log_with_features[long_log_with_features["user_idx"] == 1][["user_idx", "item_idx"]],
                    recs_file_path=None,
                )
        else:
            model.fit(train_dataset)
            model.predict_pairs(
                dataset=train_dataset,
                pairs=long_log_with_features[long_log_with_features["user_idx"] == 1][["user_idx", "item_idx"]],
                recs_file_path=path,
            )
            pred_cached = model.predict_pairs(
                dataset=train_dataset,
                pairs=long_log_with_features[long_log_with_features["user_idx"] == 1][["user_idx", "item_idx"]],
                recs_file_path=None,
            )
            pred_from_file = pd.read_parquet(path)
            assert isDataFrameEqual(pred_cached, pred_from_file)
    elif dataset_type == "polars":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        train_dataset = get_dataset_any_type(long_log_with_features)
        path = str((tmp_path / "pred.parquet").resolve().absolute())
        if model.__class__ not in client_model_list:
            with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
                model.fit(train_dataset)
                model.predict_pairs(
                    dataset=train_dataset,
                    pairs=long_log_with_features.filter(pl.col("user_idx") == 1).select("user_idx", "item_idx"),
                    recs_file_path=path,
                )
                pred_cached = model.predict_pairs(
                    dataset=train_dataset,
                    pairs=long_log_with_features.filter(pl.col("user_idx") == 1).select("user_idx", "item_idx"),
                    recs_file_path=None,
                )
        else:
            model.fit(train_dataset)
            model.predict_pairs(
                dataset=train_dataset,
                pairs=long_log_with_features.filter(pl.col("user_idx") == 1).select("user_idx", "item_idx"),
                recs_file_path=path,
            )
            pred_cached = model.predict_pairs(
                dataset=train_dataset,
                pairs=long_log_with_features.filter(pl.col("user_idx") == 1).select("user_idx", "item_idx"),
                recs_file_path=None,
            )
            pred_from_file = pd.read_parquet(path)
            isDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [
        PopRec(),
        ItemKNN(),
    ],
    ids=["pop_rec", "knn"],
)
def test_predict_to_file_spark(model, tmp_path, request):
    long_log_with_features = request.getfixturevalue("long_log_with_features")
    train_dataset = get_dataset_any_type(long_log_with_features)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit_predict(train_dataset, k=10, recs_file_path=path)
    pred_cached = model.predict(train_dataset, k=10, recs_file_path=None)
    pred_from_file = pd.read_parquet(path)
    isDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.core
@pytest.mark.parametrize(
    "model",
    [
        PopRec,
        ItemKNN,
    ],
    ids=["pop_rec", "knn"],
)
@pytest.mark.parametrize("dataset_type", ["spark", "pandas", "polars"], ids=["spark", "pandas", "polars"])
def test_predict_to_file_core(model, dataset_type, tmp_path, request):
    model = model()
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        train_dataset = get_dataset_any_type(long_log_with_features)
        path = str((tmp_path / "pred.parquet").resolve().absolute())
        if model.__class__ not in client_model_list:
            with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
                model.fit_predict(train_dataset, k=10, recs_file_path=path)
        else:
            model.fit_predict(train_dataset, k=10, recs_file_path=path)
            pred_cached = model.predict(train_dataset, k=10, recs_file_path=None)
            pred_from_file = pd.read_parquet(path)
            assert isDataFrameEqual(pred_cached, pred_from_file)
    elif dataset_type == "polars":
        long_log_with_features = request.getfixturevalue("long_log_with_features_" + dataset_type)
        train_dataset = get_dataset_any_type(long_log_with_features)
        path = str((tmp_path / "pred.parquet").resolve().absolute())
        if model.__class__ not in client_model_list:
            with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
                model.fit_predict(train_dataset, k=10, recs_file_path=path)
        else:
            model.fit_predict(train_dataset, k=10, recs_file_path=path)
            pred_cached = model.predict(train_dataset, k=10, recs_file_path=None)
            pred_from_file = pd.read_parquet(path)
            assert isDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [PopRec, UCB],
    ids=["pop_rec", "ucb"],
)
@pytest.mark.parametrize("add_cold_items", [True, False])
@pytest.mark.parametrize("predict_cold_only", [True, False])
def test_add_cold_items_for_nonpersonalized_spark(model, add_cold_items, predict_cold_only, request):
    model = model() if model == PopRec else model(sample=True)
    num_warm = 5
    # k is greater than the number of warm items to check if
    # the cold items are presented in prediction
    long_log_with_features = request.getfixturevalue("long_log_with_features")
    k = 6
    log = (
        long_log_with_features
        if not isinstance(model, (Wilson, UCB))
        else long_log_with_features.withColumn("relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1))
    )
    train_log = log.filter(sf.col("item_idx") < num_warm)
    train_dataset = get_dataset_any_type(train_log)
    model.fit(train_dataset)
    # ucb always adds cold items to prediction
    if not isinstance(model, UCB):
        model.add_cold_items = add_cold_items
    items = log.select("item_idx").distinct()
    if predict_cold_only:
        items = items.filter(sf.col("item_idx") >= num_warm)

    pred_dataset = get_dataset_any_type(log.filter(sf.col("item_idx") < num_warm))
    pred = model.predict(
        dataset=pred_dataset,
        queries=[1],
        items=items,
        k=k,
        filter_seen_items=False,
    )
    if add_cold_items or isinstance(model, UCB):
        assert pred.count() == min(k, items.count())
        if predict_cold_only:
            assert pred.select(sf.min("item_idx")).collect()[0][0] >= num_warm
            # for RandomRec relevance of an item is equal to its inverse position in the list
            if not isinstance(model, RandomRec):
                assert pred.select("relevance").distinct().count() == 1
    else:
        if predict_cold_only:
            assert pred.count() == 0
        else:
            # ucb always adds cold items to prediction
            assert pred.select(sf.max("item_idx")).collect()[0][0] < num_warm
            assert pred.count() == min(
                k,
                train_log.select("item_idx").distinct().join(items, on="item_idx").count(),
            )


@pytest.mark.core
@pytest.mark.parametrize(
    "model",
    [PopRec, UCB],
    ids=["pop_rec", "ucb"],
)
@pytest.mark.parametrize("add_cold_items", [True, False])
@pytest.mark.parametrize("predict_cold_only", [True, False])
def test_add_cold_items_for_nonpersonalized_pandas(model, add_cold_items, predict_cold_only, request):
    model = model() if model == PopRec else model(sample=True)
    num_warm = 5
    # k is greater than the number of warm items to check if
    # the cold items are presented in prediction
    long_log_with_features = request.getfixturevalue("long_log_with_features_pandas")
    k = 6
    if not isinstance(model, (Wilson, UCB)):
        long_log_with_features["relevance"] = np.where(long_log_with_features["relevance"] < 3, 0, 1)
        log = long_log_with_features
    else:
        log = long_log_with_features
    train_log = log[log["item_idx"] < num_warm]
    train_dataset = get_dataset_any_type(train_log)
    if model.__class__ not in client_model_list:
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
            model.fit(train_dataset)
    else:
        model.fit(train_dataset)
        # ucb always adds cold items to prediction
        if not isinstance(model, UCB):
            model.add_cold_items = add_cold_items
        items = pd.DataFrame(log["item_idx"].unique(), columns=["item_idx"])
        if predict_cold_only:
            items = items[items["item_idx"] >= num_warm]

        pred_dataset = get_dataset_any_type(log[log["item_idx"] < num_warm])
        pred = model.predict(
            dataset=pred_dataset,
            queries=[1],
            items=items,
            k=k,
            filter_seen_items=False,
        )
        if add_cold_items or isinstance(model, UCB):
            assert pred.shape[0] == min(k, items.shape[0])
            if predict_cold_only:
                assert pred["item_idx"].min().item() >= num_warm
                # for RandomRec relevance of an item is equal to its inverse position in the list
                if not isinstance(model, RandomRec):
                    assert pred["relevance"].unique().shape[0] == 1
        else:
            if predict_cold_only:
                assert pred.shape[0] == 0
            else:
                # ucb always adds cold items to prediction
                assert pred["item_idx"].max().item() < num_warm
                assert pred.shape[0] == min(
                    k,
                    pd.DataFrame(train_log["item_idx"].unique(), columns=["item_idx"])
                    .join(items, on="item_idx", rsuffix="_right")
                    .shape[0],
                )


@pytest.mark.core
@pytest.mark.parametrize(
    "model",
    [PopRec, UCB],
    ids=["pop_rec", "ucb"],
)
@pytest.mark.parametrize("add_cold_items", [True, False])
@pytest.mark.parametrize("predict_cold_only", [True, False])
def test_add_cold_items_for_nonpersonalized_polars(model, add_cold_items, predict_cold_only, request):
    model = model() if model == PopRec else model(sample=True)
    num_warm = 5
    # k is greater than the number of warm items to check if
    # the cold items are presented in prediction
    long_log_with_features = request.getfixturevalue("long_log_with_features_polars")
    k = 6
    if not isinstance(model, (Wilson, UCB)):
        long_log_with_features.with_columns(pl.when(pl.col("relevance") < 3).then(0).otherwise(1).alias("relevance"))
        log = long_log_with_features
    else:
        log = long_log_with_features
    train_log = log.filter(pl.col("item_idx") < num_warm)
    train_dataset = get_dataset_any_type(train_log)
    if model.__class__ not in client_model_list:
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute"):
            model.fit(train_dataset)
    else:
        model.fit(train_dataset)
        # ucb always adds cold items to prediction
        if not isinstance(model, UCB):
            model.add_cold_items = add_cold_items
        items = log.select("item_idx").unique()
        if predict_cold_only:
            items = items.filter(pl.col("item_idx") >= num_warm)

        pred_dataset = get_dataset_any_type(log.filter(pl.col("item_idx") < num_warm))
        pred = model.predict(
            dataset=pred_dataset,
            queries=[1],
            items=items,
            k=k,
            filter_seen_items=False,
        )
        if add_cold_items or isinstance(model, UCB):
            assert pred.height == min(k, items.height)
            if predict_cold_only:
                assert pred.select("item_idx").min().item() >= num_warm
                # for RandomRec relevance of an item is equal to its inverse position in the list
                if not isinstance(model, RandomRec):
                    assert pred.select("relevance").unique().height == 1
        else:
            if predict_cold_only:
                assert pred.height == 0
            else:
                # ucb always adds cold items to prediction
                assert pred.select("item_idx").max().item() < num_warm
                assert pred.height == min(
                    k,
                    train_log.select("item_idx").unique().join(items, on="item_idx").height,
                )


"""
@pytest.mark.core           # TODO: решить, что с ним делать? Когда модель еще не выставлена
@pytest.mark.parametrize(
    "model",
    [PopRec(), UCB()],
    ids=["pop_rec", "ucb"],
)
@pytest.mark.parametrize("params", [{"test_param": 42}, {"another_param": "extra_value"}])
def test_set_params(model, params):
    model.set_params(**params)
    for key, value in params.items():
        assert getattr(model, key) == value
"""


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_calc_fill(base_model, arguments, datasets):
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    fill_pd = model_pd._impl._calc_fill(model_pd.item_popularity, model_pd.cold_weight, model_pd.rating_column)
    fill_pl = model_pl._impl._calc_fill(model_pl.item_popularity, model_pl.cold_weight, model_pl.rating_column)
    fill_spark = model_spark._impl._calc_fill(
        model_spark.item_popularity, model_spark.cold_weight, model_spark.rating_column
    )
    assert fill_pd == fill_spark, "Pandas fill not equals Spark fill"
    assert fill_pl == fill_spark, "Polars fill not equals Spark fill"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_get_selected_item_popularity(base_model, arguments, datasets, spark):
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    items_pd = pd.DataFrame({model_pd.item_column: [3]})
    items_pl = pl.DataFrame({model_pl.item_column: [3]})
    items_spark = spark.createDataFrame([{model_spark.item_column: 3}])
    item_popularity_pl = model_pl._impl._get_selected_item_popularity(items_pl)
    item_popularity_pd = model_pd._impl._get_selected_item_popularity(items_pd)
    item_popularity_spark = model_spark._impl._get_selected_item_popularity(items_spark)
    assert isDataFrameEqual(
        item_popularity_pd, item_popularity_spark
    ), "Pandas item_popularity not equals Spark item_popularity"
    assert isDataFrameEqual(
        item_popularity_pd, item_popularity_pl
    ), "Pandas item_popularity not equals Polars item_popularity"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_calc_max_hist_len(base_model, arguments, datasets, spark):
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    queries_pd = pd.DataFrame({model_pd.query_column: [2]})
    queries_pl = pl.DataFrame({model_pl.query_column: [2]})
    queries_spark = spark.createDataFrame([{model_spark.query_column: 2}])
    max_hist_len_pd = model_pd._impl._calc_max_hist_len(pandas_df, queries_pd)
    max_hist_len_pl = model_pl._impl._calc_max_hist_len(polars_df, queries_pl)
    max_hist_len_spark = model_spark._impl._calc_max_hist_len(spark_df, queries_spark)
    assert max_hist_len_pd == max_hist_len_spark, "Pandas max_hist_len not equals Spark max_hist_len"
    assert max_hist_len_pd == max_hist_len_pl, "Pandas max_hist_len not equals Spark max_hist_len"
    queries_pd = pd.DataFrame({model_pd.query_column: [1234567]})
    queries_pl = pl.DataFrame({model_pl.query_column: [1234567]})
    queries_spark = spark.createDataFrame([{model_spark.query_column: 1234567}])
    max_hist_len_pd = model_pd._impl._calc_max_hist_len(pandas_df, queries_pd)
    max_hist_len_pl = model_pl._impl._calc_max_hist_len(polars_df, queries_pl)
    max_hist_len_spark = model_spark._impl._calc_max_hist_len(spark_df, queries_spark)
    assert max_hist_len_pd == max_hist_len_spark, "Pandas max_hist_len not equals Spark max_hist_len"
    assert max_hist_len_pd == max_hist_len_pl, "Pandas max_hist_len not equals Spark max_hist_len"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_calc_max_hist_len(base_model, arguments, datasets, spark):
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    queries_pd = pd.DataFrame({model_pd.query_column: [2]})
    queries_pl = pl.DataFrame({model_pl.query_column: [2]})
    queries_spark = spark.createDataFrame([{model_spark.query_column: 2}])
    max_hist_len_pd = model_pd._impl._calc_max_hist_len(pandas_df, queries_pd)
    max_hist_len_pl = model_pl._impl._calc_max_hist_len(polars_df, queries_pl)
    max_hist_len_spark = model_spark._impl._calc_max_hist_len(spark_df, queries_spark)
    assert max_hist_len_pd == max_hist_len_spark, "Pandas max_hist_len not equals Spark max_hist_len"
    assert max_hist_len_pd == max_hist_len_pl, "Pandas max_hist_len not equals Spark max_hist_len"
    queries_pd = pd.DataFrame({model_pd.query_column: [1234567]})
    queries_pl = pl.DataFrame({model_pl.query_column: [1234567]})
    queries_spark = spark.createDataFrame([{model_spark.query_column: 1234567}])
    max_hist_len_pd = model_pd._impl._calc_max_hist_len(pandas_df, queries_pd)
    max_hist_len_pl = model_pl._impl._calc_max_hist_len(polars_df, queries_pl)
    max_hist_len_spark = model_spark._impl._calc_max_hist_len(spark_df, queries_spark)
    assert max_hist_len_pd == max_hist_len_spark, "Pandas max_hist_len not equals Spark max_hist_len"
    assert max_hist_len_pd == max_hist_len_pl, "Pandas max_hist_len not equals Spark max_hist_len"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_filter_interactions_queries_items_dataframes(base_model, arguments, datasets):
    query_features_pl, item_features_pl, dataset_pl = (
        datasets["polars"].query_features,
        datasets["polars"].item_features,
        datasets["polars"],
    )
    query_features_pd, item_features_pd, dataset_pd = (
        datasets["pandas"].query_features,
        datasets["pandas"].item_features,
        datasets["pandas"],
    )
    query_features_spark, item_features_spark, dataset_spark = (
        datasets["spark"].query_features,
        datasets["spark"].item_features,
        datasets["spark"],
    )
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(dataset_pd)
    model_pl.fit(dataset_pl)
    model_spark.fit(dataset_spark)

    dataset_pd, queries_pd, items_pd = model_pd._impl._filter_interactions_queries_items_dataframes(
        None, 1, queries=query_features_pd, items=item_features_pd
    )
    dataset_pl, queries_pl, items_pl = model_pl._impl._filter_interactions_queries_items_dataframes(
        None, 1, queries=query_features_pl, items=item_features_pl
    )
    dataset_spark, queries_spark, items_spark = model_spark._impl._filter_interactions_queries_items_dataframes(
        None, 1, queries=query_features_spark, items=item_features_spark
    )
    if dataset_pd is not None and dataset_pd.interactions is not None:
        assert isDataFrameEqual(dataset_pd.interactions, dataset_spark.interactions), "Pandas df not equals Spark df"
        assert isDataFrameEqual(dataset_pd.interactions, dataset_pl.interactions), "Pandas df not equals Polars df"
    elif dataset_pd is not None and dataset_pd.interactions is None:
        assert dataset_pd.interactions == dataset_pl.interactions == dataset_spark.interactions
    else:
        assert dataset_pd == dataset_pl == dataset_spark
    if queries_pd is not None:
        assert isDataFrameEqual(queries_pd, queries_pl), "Pandas df not equals Polars df"
        assert isDataFrameEqual(queries_pd, queries_spark), "Pandas df not equals Spark df"
    else:
        assert queries_pd == queries_pl == queries_spark

    if items_pd is not None:
        assert isDataFrameEqual(items_pl, items_pl), "Pandas df not equals Spark df"
        assert isDataFrameEqual(items_pd, items_pl), "Pandas df not equals Polars df"
    else:
        assert items_pl == items_pd == items_spark


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_fit_predict_all_arguments(base_model, arguments, datasets, tmp_path):
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    path_pd = str((tmp_path / "pred_pd.parquet").resolve().absolute())
    path_pl = str((tmp_path / "pred_pl.parquet").resolve().absolute())
    path_spark = str((tmp_path / "pred_spark.parquet").resolve().absolute())
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pl.fit_predict(polars_df, 5, [0], [3], filter_seen_items=False, recs_file_path=path_pl)
    model_spark.fit_predict(spark_df, 5, [0], [3], filter_seen_items=False, recs_file_path=path_spark)
    model_pd.fit_predict(pandas_df, 5, [0], [3], filter_seen_items=False, recs_file_path=path_pd)
    pred_pl = pd.read_parquet(path_pl)
    pred_spark = pd.read_parquet(path_spark)
    pred_pd = pd.read_parquet(path_pd)
    assert isDataFrameEqual(pred_pd, pred_spark), "Pandas preds not equals Spark preds"
    assert isDataFrameEqual(pred_pd, pred_pl), "Pandas preds not equals Polars preds"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_predict_pairs_incorrect_call(base_model, arguments, datasets, spark):
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)

    pairs_pd = pd.DataFrame({cols[0]: [1, 2], cols[1]: [2, 3]})
    pairs_pl = pl.DataFrame({cols[0]: [1, 2], cols[1]: [2, 3]})
    pairs_spark = spark.createDataFrame([{cols[0]: 1, cols[1]: 3}])
    with pytest.raises(NotFittedModelError):
        model_pl.predict_pairs(pairs_pl, dataset=polars_df)
    with pytest.raises(NotFittedModelError):
        model_spark.predict_pairs(pairs_spark, dataset=spark_df)
    with pytest.raises(NotFittedModelError):
        model_pd.predict_pairs(pairs_pd, dataset=pandas_df)

    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    pairs_pd = pd.DataFrame({"wrong": [1], "another": [2]})
    pairs_pl = pl.DataFrame({"wrong": [1], "another": [2]})
    pairs_spark = spark.createDataFrame([{"wrong": 1, "another": 2}])
    with pytest.raises(ValueError, match="pairs must be a dataframe with columns strictly"):
        model_pl.predict_pairs(pairs_pl, dataset=polars_df)
    with pytest.raises(ValueError, match="pairs must be a dataframe with columns strictly"):
        model_spark.predict_pairs(pairs_spark, dataset=spark_df)
    with pytest.raises(ValueError, match="pairs must be a dataframe with columns strictly"):
        model_pd.predict_pairs(pairs_pd, dataset=pandas_df)


"""
@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_filter_seen(base_model, arguments, datasets):
    dataset_pl = datasets["polars"]
    dataset_pd = datasets["pandas"]
    dataset_spark = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(dataset_pd)
    model_pl.fit(dataset_pl)
    model_spark.fit(dataset_spark)
    dataset_pd, queries_pd, items_pd = model_pd._impl._filter_interactions_queries_items_dataframes(
        dataset_pd, 1
    )
    dataset_pl, queries_pl, items_pl = model_pl._impl._filter_interactions_queries_items_dataframes(
        dataset_pl, 1
    )
    dataset_spark, queries_spark, items_spark = model_spark._impl._filter_interactions_queries_items_dataframes(
        dataset_spark, 1
    )
    recs_pd = model_pd._impl._predict_without_sampling(dataset_pd, 1, queries_pd, items_pd).head(0)
    recs_pl = model_pl._impl._predict_without_sampling(dataset_pl, 1, queries_pl, items_pl).limit(1)
    recs_spark = model_spark._impl._predict_without_sampling(dataset_spark, 1, queries_spark, items_spark).limit(0)
    recs_pd = model_pd._impl._filter_seen(recs=recs_pd, interactions=dataset_pd.interactions, queries=queries_pd, k=1)
    recs_pl = model_pl._impl._filter_seen(recs=recs_pl, interactions=dataset_pl.interactions, queries=queries_pl, k=1)
    recs_spark = model_spark._impl._filter_seen(
        recs=recs_spark, interactions=dataset_spark.interactions, queries=queries_spark, k=1
    )
    assert recs_pd.empty == recs_pl.is_empty() == recs_spark.limit(1).isEmpty()
"""


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (PopRec, {"use_rating": True})],
    ids=["pop_rec", "pop_rec_with_rating"],
)
def test_fit_predict_the_same_framework_spark(base_model, arguments, datasets, big_datasets):
    for dataset in [datasets, big_datasets]:
        results = {}
        for framework, df in dataset.items():
            model = base_model(**arguments)
            res = None
            if framework == "pandas":
                res = model.fit_predict(df, k=1).sort_values(["user_id", "item_id"])
            elif framework == "spark":
                res = model.fit_predict(df, k=1).sort("user_id", "item_id").toPandas()
            if res is not None:
                results.update({f"{framework}": res})
            del model

        pandas_res = results["pandas"]
        spark_res = results["spark"]
        assert isDataFrameEqual(pandas_res, spark_res), "Dataframes are not equals"


@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (PopRec, {"use_rating": True})],
    ids=["pop_rec", "pop_rec_with_rating"],
)
def test_fit_predict_the_same_framework_polars(base_model, arguments, datasets, big_datasets):
    for dataset in [datasets, big_datasets]:
        results = {}
        for framework, df in dataset.items():
            model = base_model(**arguments)
            res = None
            if framework == "pandas":
                res = model.fit_predict(df, k=1).sort_values(["user_id", "item_id"])
            elif framework == "polars":
                res = model.fit_predict(df, k=1).sort("user_id", "item_id").to_pandas()
            if res is not None:
                results.update({f"{framework}": res})
            del model

        pandas_res = results["pandas"]
        polars_res = results["polars"]
        assert isDataFrameEqual(pandas_res, polars_res), "Dataframes are not equals"


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
def test_fit_predict_different_frameworks_spark(base_model, arguments, predict_framework, datasets, big_datasets):
    for dataset in [datasets, big_datasets]:
        results = {}
        model_default = base_model(**arguments)
        base_res = model_default.fit_predict(dataset["spark"], k=1).sort("user_id", "item_id").toPandas()
        for train_framework, df in dataset.items():
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
            elif predict_framework == "polars":
                model.to_polars()
                df.to_polars()
                res
            if res is not None:
                results.update({f"{train_framework}_{predict_framework}": res})
        cnt = 0
        for type_of_convertation, dataframe in results.items():
            cnt += 1
            assert isDataFrameEqual(
                base_res, dataframe
            ), f"Not equal dataframes in {type_of_convertation} pair of train-predict"


@pytest.mark.core
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (PopRec, {"use_rating": True})],
    ids=["pop_rec", "pop_rec_with_rating"],
)
def test_fit_predict_different_frameworks_pandas_polars(base_model, arguments, datasets, big_datasets):
    for dataset in [datasets, big_datasets]:
        polars_df = dataset["polars"]
        pandas_df = dataset["pandas"]
        model = base_model(**arguments)
        model.fit(pandas_df)
        model.to_polars()
        res1 = model.predict(polars_df, k=1).sort("user_id", "item_id").to_pandas()
        model = base_model(**arguments)
        model.fit(polars_df)
        model.to_pandas()
        res2 = model.predict(pandas_df, k=1).sort_values(["user_id", "item_id"])
        assert isDataFrameEqual(res1, res2), "Not equal dataframes in pair of train-predict"
