import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.metrics import NDCG
from replay.models import UCB, ClusterRec, ItemKNN, PopRec, RandomRec, Wilson, client_model_list
from replay.models.base_rec_client import DataModelMissmatchError, NotFittedModelError
from replay.utils.common import convert2polars, convert2spark
from replay.utils.types import DataFrameLike
from tests.models.conftest import cols
from tests.utils import SparkDataFrame, get_dataset_any_type, isDataFrameEqual

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf

SEED = 123


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
        pairs_pred_k_pd = model.predict_pairs(pairs, train_dataset, k=1)
        pairs_pred_pd = model.predict_pairs(pairs, train_dataset, k=None)
        counts_k_pd = pairs_pred_k_pd.groupby("user_idx").size()
        counts_pd = pairs_pred_pd.groupby("user_idx").size()
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
        pairs_pred_k_pl = model.predict_pairs(pairs, train_dataset, k=1)
        pairs_pred_pl = model.predict_pairs(pairs, train_dataset, k=None)
        counts_k_pl = pairs_pred_k_pl.group_by("user_idx").count()
        counts_pl = pairs_pred_pl.group_by("user_idx").count()
        assert all(counts_k_pl["count"].to_pandas().to_numpy() <= 1)
        assert any(counts_pl["count"].to_pandas().to_numpy() > 1)

    if model.__class__ in client_model_list:
        assert isDataFrameEqual(pairs_pred_k_pd, pairs_pred_k_spark), "Pandas predictions not equals Spark predictions"
        # Code below like assert isDataFrameEqual(pairs_pred_k_pd, pairs_pred_k_pl)
        for index_pd, value_pd in pairs_pred_k_pd["relevance"].items():
            for index_pl, value_pl in pairs_pred_k_pl.select("relevance").to_pandas()["relevance"].items():
                if index_pd == index_pl:
                    assert value_pl, value_pd == value_pl
        assert pd.concat([pairs_pred_k_pd, pairs_pred_k_pl.to_pandas()]).drop_duplicates(keep=False).shape[0] == 0
        assert all(pairs_pred_k_pd.columns == pairs_pred_k_pl.columns)
        assert isDataFrameEqual(pairs_pred_pd, pairs_pred_spark), "Pandas predictions not equals Spark predictions"
        # Code below like assert isDataFrameEqual(pairs_pred_pd, pairs_pred_pl),
        for index_pd, value_pd in pairs_pred_pd["relevance"].items():
            for index_pl, value_pl in pairs_pred_pl.select("relevance").to_pandas()["relevance"].items():
                if index_pd == index_pl:
                    assert value_pl, value_pd == value_pl
        assert pd.concat([pairs_pred_pd, pairs_pred_pl.to_pandas()]).drop_duplicates(keep=False).shape[0] == 0


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
        PopRec(),
    ],
    ids=["pop_rec"],
)
def test_predict_before_fit(model, request):
    log = request.getfixturevalue("log")
    empty_df = log
    pred_dataset = get_dataset_any_type(empty_df)
    with pytest.raises(NotFittedModelError):
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
def test_filter_seen_items_spark(request):
    log = request.getfixturevalue("log")
    model = PopRec()
    train_dataset = get_dataset_any_type(log.filter((sf.col("user_idx") != 0) | (sf.col("user_idx").isNull())))
    pred_dataset = get_dataset_any_type(log)
    model.fit(train_dataset)
    pred = model.predict(dataset=pred_dataset, queries=[3], k=5)
    assert pred.count() == 2
    pred = model.predict(dataset=pred_dataset, queries=[0], k=5)
    assert pred.count() == 1
    pred = model.predict(dataset=pred_dataset, queries=[0], k=5, filter_seen_items=False)
    assert pred.count() == 4
    with pytest.raises(DataModelMissmatchError, match="calculate input data due to missmatch of types"):
        pred = model.predict(dataset=pred_dataset, queries=1, k=5)


@pytest.mark.core
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"], ids=["pandas", "polars"])
def test_filter_seen_items_core(dataset_type, request):
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
        # ne_missing keeps null as pandas if !=0
        train_dataset = get_dataset_any_type(log.filter(pl.col("user_idx").ne_missing(pl.lit(0))))
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
    [PopRec()],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("df_type", ["", "none_items", "random_sorted", "none_queries", "one_query"])
def test_predict_new_queries_spark(model, df_type, request):
    if df_type != "":
        df_type = "_and_" + df_type
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type)
    user_features = request.getfixturevalue("user_features")
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter((sf.col("user_idx") != 0) | (sf.col("user_idx").isNull())),
        inf_log=long_log_with_features,
        user_features=user_features.drop("gender"),
        queries=[0],
    )
    if model.__class__ in client_model_list:
        assert pred.count() == 1
        assert pred.collect()[0][0] == 0


@pytest.mark.core
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"], ids=["pandas", "polars"])
@pytest.mark.parametrize(
    "model",
    [ItemKNN(), PopRec()],
    ids=["knn", "pop_rec"],
)
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_predict_new_queries_core(dataset_type, df_type, model, request):
    if df_type != "":
        df_type = "_and_" + df_type
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
        user_features = request.getfixturevalue("user_features_" + dataset_type)
        pred = fit_predict_selected(
            model,
            train_log=long_log_with_features[long_log_with_features["user_idx"] != 0],
            inf_log=long_log_with_features,
            user_features=user_features.drop("gender", axis=1),
            queries=[0],
        )
        if pred is not None:
            assert pred.shape[0] == 1
            assert pred.iloc[0, 0] == 0
        else:
            assert model.__class__ not in client_model_list
    elif dataset_type == "polars":
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
        user_features = request.getfixturevalue("user_features_" + dataset_type)
        pred = fit_predict_selected(
            model,
            # ne_missing keeps null as pandas if !=0
            train_log=long_log_with_features.filter(pl.col("user_idx").ne_missing(pl.lit(0))),
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_predict_cold_spark(model, df_type, request):
    if df_type != "":
        df_type = "_and_" + df_type
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type)
    user_features = request.getfixturevalue("user_features")
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter((sf.col("user_idx") != 0) | (sf.col("user_idx").isNull())),
        inf_log=long_log_with_features.filter((sf.col("user_idx") != 0) | (sf.col("user_idx").isNull())),
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_predict_cold_core(dataset_type, df_type, model, request):
    if df_type != "":
        df_type = "_and_" + df_type
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
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
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
        user_features = request.getfixturevalue("user_features_" + dataset_type)
        pred = fit_predict_selected(
            model,
            # ne_missing keeps null as pandas if col !=0
            train_log=long_log_with_features.filter(pl.col("user_idx").ne_missing(pl.lit(0))),
            inf_log=long_log_with_features.filter(pl.col("user_idx").ne_missing(pl.lit(0))),
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
    "base_model, arguments", [(PopRec, {})], ids=["pop_rec"]  # , (ItemKNN, {})],  # , "item_knn"],
)
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_fit_predict_item_features(base_model, df_type, arguments, request):
    if df_type != "":
        df_type = "_and_" + df_type
    dataset_type = "spark"
    model = base_model(**arguments)
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type)
    item_features = request.getfixturevalue("item_features")
    pred_spark = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter((sf.col("user_idx") != 0) | (sf.col("user_idx").isNull())),
        inf_log=long_log_with_features,
        item_features=item_features.drop("color", "class"),
        queries=[0],
    )
    if "quer" in df_type:
        pass
    elif pred_spark is not None:
        assert pred_spark.count() == 1
        assert pred_spark.collect()[0][0] == 0
    else:
        assert model.__class__ not in client_model_list
    dataset_type = "pandas"
    model = base_model(**arguments)
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
    item_features = request.getfixturevalue("item_features_" + dataset_type)
    pred_pd = fit_predict_selected(
        model,
        train_log=long_log_with_features[long_log_with_features["user_idx"] != 0],
        inf_log=long_log_with_features,
        item_features=item_features.drop(["color", "class"], axis=1),
        queries=[0],
    )
    if "quer" in df_type:
        pass
    elif pred_pd is not None:
        assert pred_pd.shape[0] == 1
        assert pred_pd.iloc[0, 0] == 0
    else:
        assert model.__class__ not in client_model_list
    dataset_type = "polars"
    model = base_model(**arguments)
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
    item_features = request.getfixturevalue("item_features_" + dataset_type)
    pred_pl = fit_predict_selected(
        model,
        # ne_missing keeps null as pandas if col !=0
        train_log=long_log_with_features.filter(
            pl.col("user_idx").ne_missing(pl.lit(0))
        ),  # keeps null as pandas if !=0
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_predict_pairs_to_file_spark(model, df_type, tmp_path, request):
    if df_type != "":
        df_type = "_and_" + df_type
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"])
def test_predict_pairs_to_file_core(model, dataset_type, df_type, tmp_path, request):
    if df_type != "":
        df_type = "_and_" + df_type
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
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
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_predict_to_file_spark(model, df_type, tmp_path, request):
    if df_type != "":
        df_type = "_and_" + df_type
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
@pytest.mark.parametrize("dataset_type", ["pandas", "polars"])
def test_predict_to_file_core(model, dataset_type, df_type, tmp_path, request):
    if df_type != "":
        df_type = "_and_" + df_type
    model = model()
    if dataset_type == "pandas":
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
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
        long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_" + dataset_type)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_add_cold_items_for_nonpersonalized_spark(model, add_cold_items, df_type, predict_cold_only, request):
    if df_type != "":
        df_type = "_and_" + df_type
    model = model() if model == PopRec else model(sample=True)
    num_warm = 5
    # k is greater than the number of warm items to check if
    # the cold items are presented in prediction
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_add_cold_items_for_nonpersonalized_pandas(model, add_cold_items, df_type, predict_cold_only, request):
    if df_type != "":
        df_type = "_and_" + df_type
    model = model() if model == PopRec else model(sample=True)
    num_warm = 5
    # k is greater than the number of warm items to check if
    # the cold items are presented in prediction
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_pandas")
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
            queries=[0],
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_add_cold_items_for_nonpersonalized_polars(model, add_cold_items, df_type, predict_cold_only, request):
    if df_type != "":
        df_type = "_and_" + df_type
    model = model() if model == PopRec else model(sample=True)
    num_warm = 5
    # k is greater than the number of warm items to check if
    # the cold items are presented in prediction
    long_log_with_features = request.getfixturevalue("long_log_with_features" + df_type + "_polars")
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
            queries=[0],
            items=items,
            k=k,
            filter_seen_items=False,
        )
        if add_cold_items or isinstance(model, UCB):
            assert pred.height == min(k, items.drop_nulls().height)
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


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize(
    "params",
    [
        {"add_cold_items": True},
        {"model": 123},
        {"study": 123},
        {"criterion": NDCG(1)},
        {"cold_weight": 0.6},
        {"fill": 0.4},
        {"fit_queries": pd.DataFrame([1, 2, 3], columns=["user_id"])},
        {"fit_items": pd.DataFrame([1, 2, 3], columns=["item_id"])},
        {"item_popularity": pd.DataFrame([1, 2, 3], columns=["item_popularity"])},
    ],
    ids=[
        "add_cold_items",
        "model",
        "study",
        "criterion",
        "cold_weight",
        "fill",
        "fit_queries",
        "fit_items",
        "item_popularity",
    ],
)
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_set_params(base_model, arguments, params, df_type, request):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    with pytest.raises(AttributeError, match=r"does not have the 'set_params\(\)' method"):
        model_pd.set_params(**params)
    with pytest.raises(AttributeError, match=r"does not have the 'set_params\(\)' method"):
        model_pl.set_params(**params)
    with pytest.raises(AttributeError, match=r"does not have the 'set_params\(\)' method"):
        model_spark.set_params(**params)
    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    model_pd.set_params(**params)
    for key, value in params.items():
        if isinstance(value, pd.DataFrame):
            params[key] = convert2polars(value)
    model_pl.set_params(**params)
    for key, value in params.items():
        if isinstance(value, pd.DataFrame):
            params[key] = convert2spark(value)
    if next(iter(params.keys())) == "item_popularity":
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute 'unpersist'"):
            model_spark.set_params(**params)
    else:
        model_spark.set_params(**params)

    for key, value in params.items():
        if isinstance(value, tuple(DataFrameLike.__args__)):
            assert isDataFrameEqual(getattr(model_spark, key), value)
        else:
            assert getattr(model_spark, key) == value
        if isinstance(value, tuple(DataFrameLike.__args__)):
            assert isDataFrameEqual(getattr(model_pd, key), value)
        else:
            assert getattr(model_pd, key) == value
        if isinstance(value, tuple(DataFrameLike.__args__)):
            assert isDataFrameEqual(getattr(model_pl, key), value)
        else:
            assert getattr(model_pl, key) == value


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_calc_fill(base_model, arguments, df_type, request):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_get_selected_item_popularity(base_model, arguments, df_type, request, spark):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    items_pd = pd.DataFrame({model_pd.item_column: [1]})
    items_pl = pl.DataFrame({model_pl.item_column: [1]})
    items_spark = spark.createDataFrame([{model_spark.item_column: 1}])
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_calc_max_hist_len(base_model, arguments, df_type, request, spark):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_filter_interactions_queries_items_dataframes(base_model, arguments, df_type, request):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
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
@pytest.mark.parametrize("filter_seen", [False, True])
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_fit_predict_all_arguments(base_model, arguments, filter_seen, df_type, request, tmp_path):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    path_pd = str((tmp_path / "pred_pd.parquet").resolve().absolute())
    path_pl = str((tmp_path / "pred_pl.parquet").resolve().absolute())
    path_spark = str((tmp_path / "pred_spark.parquet").resolve().absolute())
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pl.fit_predict(polars_df, 5, [0], [3], filter_seen_items=filter_seen, recs_file_path=path_pl)
    model_spark.fit_predict(spark_df, 5, [0], [3], filter_seen_items=filter_seen, recs_file_path=path_spark)
    model_pd.fit_predict(pandas_df, 5, [0], [3], filter_seen_items=filter_seen, recs_file_path=path_pd)
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
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_predict_pairs_incorrect_call(base_model, arguments, df_type, request, spark):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
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
    with pytest.raises(DataModelMissmatchError):
        model_pl.predict_pairs(pairs_spark, dataset=polars_df)
    with pytest.raises(DataModelMissmatchError):
        model_spark.predict_pairs(pairs_pd, dataset=spark_df)
    with pytest.raises(DataModelMissmatchError):
        model_pd.predict_pairs(pairs_pl, dataset=pandas_df)


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
def test_predict_proba(base_model, arguments, request):
    log_spark = request.getfixturevalue("log")
    spark_df = get_dataset_any_type(log_spark)
    log_pandas = request.getfixturevalue("log_pandas")
    pandas_df = get_dataset_any_type(log_pandas)
    log_polars = request.getfixturevalue("log_polars")
    polars_df = get_dataset_any_type(log_polars)
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(pandas_df)
    model_pl.fit(polars_df)
    model_spark.fit(spark_df)
    n_users, n_actions, K = 2, 5, 3
    users_spark = convert2spark(pd.DataFrame({"user_idx": np.arange(n_users)}))
    items_spark = convert2spark(pd.DataFrame({"item_idx": np.arange(n_actions)}))
    users_pd = pd.DataFrame({"user_idx": np.arange(n_users)})
    items_pd = pd.DataFrame({"item_idx": np.arange(n_actions)})
    users_pl = convert2polars(pd.DataFrame({"user_idx": np.arange(n_users)}))
    items_pl = convert2polars(pd.DataFrame({"item_idx": np.arange(n_actions)}))
    pred = model_spark._predict_proba(spark_df, K, users_spark, items_spark, False)
    assert pred.shape == (n_users, n_actions, K)
    assert np.allclose(pred.sum(1), np.ones(shape=(n_users, K)))
    with pytest.raises(NotImplementedError):
        model_pl._predict_proba(polars_df, 1, users_pl, items_pl, False)
    with pytest.raises(NotImplementedError):
        model_pd._predict_proba(pandas_df, 1, users_pd, items_pd, False)


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_predict_proba_incorrect_call(base_model, arguments, df_type, request):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
    polars_df = datasets["polars"]
    pandas_df = datasets["pandas"]
    spark_df = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    with pytest.raises(NotFittedModelError):
        model_pl._predict_proba(polars_df, 1, [0], [3])
    with pytest.raises(NotFittedModelError):
        model_spark._predict_proba(spark_df, 1, [0], [3])
    with pytest.raises(NotFittedModelError):
        model_pd._predict_proba(pandas_df, 1, [0], [3])


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("df_type", ["", "none_queries", "none_items", "random_sorted", "one_query"])
def test_filter_seen(base_model, arguments, df_type, request):
    if df_type != "":
        df_type = "_" + df_type
    datasets = request.getfixturevalue("datasets" + df_type)
    dataset_pl = datasets["polars"]
    dataset_pd = datasets["pandas"]
    dataset_spark = datasets["spark"]
    model_pd = base_model(**arguments)
    model_pl = base_model(**arguments)
    model_spark = base_model(**arguments)
    model_pd.fit(dataset_pd)
    model_pl.fit(dataset_pl)
    model_spark.fit(dataset_spark)
    dataset_pd, queries_pd, items_pd = model_pd._impl._filter_interactions_queries_items_dataframes(dataset_pd, 1)
    dataset_pl, queries_pl, items_pl = model_pl._impl._filter_interactions_queries_items_dataframes(dataset_pl, 1)
    dataset_spark, queries_spark, items_spark = model_spark._impl._filter_interactions_queries_items_dataframes(
        dataset_spark, 1
    )
    recs_pd = model_pd._impl._predict_without_sampling(dataset_pd, 1, queries_pd, items_pd).head(0)
    recs_pl = model_pl._impl._predict_without_sampling(dataset_pl, 1, queries_pl, items_pl).limit(0)
    recs_spark = model_spark._impl._predict_without_sampling(dataset_spark, 1, queries_spark, items_spark).limit(0)
    recs_pd = model_pd._impl._filter_seen(recs=recs_pd, interactions=dataset_pd.interactions, queries=queries_pd, k=1)
    recs_pl = model_pl._impl._filter_seen(recs=recs_pl, interactions=dataset_pl.interactions, queries=queries_pl, k=1)
    recs_spark = model_spark._impl._filter_seen(
        recs=recs_spark, interactions=dataset_spark.interactions, queries=queries_spark, k=1
    )
    assert recs_pd.empty == recs_pl.is_empty() == recs_spark.limit(1).isEmpty()


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [
        (PopRec, {}),
        (PopRec, {"use_rating": True}),
    ],
    ids=[
        "pop_rec",
        "pop_rec_with_rating",
    ],
)
@pytest.mark.parametrize("predict_framework", ["pandas", "polars", "spark"])
@pytest.mark.parametrize("train_framework", ["pandas", "polars", "spark"])
@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "datasets",
        "big_datasets",
        "datasets_none_items",
        "datasets_none_queries",
        "datasets_one_query",
        "datasets_random_sorted",
    ],
)
def test_fit_predict_different_frameworks_spark(
    base_model, arguments, train_framework, predict_framework, dataset_fixture, request
):
    dataset = request.getfixturevalue(dataset_fixture)
    model_default = base_model().__class__(**arguments)
    base_res = model_default.fit_predict(dataset["spark"], k=1).toPandas()
    df = dataset[train_framework]
    if predict_framework == train_framework:
        return
    model = base_model().__class__(**arguments)
    model.fit(df)
    if predict_framework == "pandas":
        model.to_pandas()
        df.to_pandas()
        res = model.predict(df, k=1)
    elif predict_framework == "spark":
        model.to_spark()
        df.to_spark()
        res = model.predict(df, k=1).toPandas()
    elif predict_framework == "polars":
        model.to_polars()
        df.to_polars()
        res = model.predict(df, k=1).to_pandas()
    if res is not None and base_res is not None:
        assert isDataFrameEqual(
            base_res, res
        ), f"Not equal dataframes in {train_framework}_{predict_framework} pair of train-predict"
    else:
        assert base_res == res


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (PopRec, {"use_rating": True})],
    ids=["pop_rec", "pop_rec_with_rating"],
)  # TODO: Not equal dfs on datasets_none_items fixture. Fix it
@pytest.mark.parametrize(
    "dataset_fixture",
    ["datasets", "big_datasets", "datasets_none_queries", "datasets_one_query", "datasets_random_sorted"],
)
def test_fit_predict_different_frameworks_pandas_polars(base_model, arguments, dataset_fixture, request):
    dataset = request.getfixturevalue(dataset_fixture)
    polars_df = dataset["polars"]
    pandas_df = dataset["pandas"]
    model = base_model().__class__(**arguments)
    model.fit(pandas_df)
    model.to_polars()
    res1 = model.predict(polars_df, k=1).to_pandas()
    model = base_model().__class__(**arguments)
    model.fit(polars_df)
    model.to_pandas()
    res2 = model.predict(pandas_df, k=1)
    assert isDataFrameEqual(res1, res2), "Not equal dataframes in pair of train-predict"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {}), (PopRec, {"use_rating": True})],
    ids=["pop_rec", "pop_rec_with_rating"],
)
@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "datasets",
        "big_datasets",
        "datasets_none_items",
        "datasets_none_queries",
        "datasets_one_query",
        "datasets_random_sorted",
    ],
)
def test_fit_predict_the_same_framework(base_model, arguments, dataset_fixture, request):
    dataset = request.getfixturevalue(dataset_fixture)
    df_pd = dataset["pandas"]
    model_pd = base_model().__class__(**arguments)
    pandas_res = model_pd.fit_predict(df_pd, k=1)
    df_pl = dataset["polars"]
    model_pl = base_model().__class__(**arguments)
    polars_res = model_pl.fit_predict(df_pl, k=1).to_pandas()
    df_spark = dataset["spark"]
    model_spark = base_model().__class__(**arguments)
    spark_res = model_spark.fit_predict(df_spark, k=1).toPandas()
    assert isDataFrameEqual(pandas_res, polars_res), "Pandas results are not equals Polars results"
    assert isDataFrameEqual(pandas_res, spark_res), "Pandas results are not equals Spark results"


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, arguments",
    [(PopRec, {})],
    ids=["pop_rec"],
)
@pytest.mark.parametrize("type_of_impl", ["pandas", "spark", "polars"])
@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "datasets",
        "big_datasets",
        "datasets_none_items",
        "datasets_none_queries",
        "datasets_one_query",
        "datasets_random_sorted",
    ],
)
def test_compare_fit_predict_client_and_implementation(base_model, arguments, type_of_impl, dataset_fixture, request):
    datasets = request.getfixturevalue(dataset_fixture)
    dataset = datasets[type_of_impl]
    model_client = base_model(**arguments)
    model_impl = base_model(**arguments)
    res_client = model_client.fit_predict(dataset, k=2)

    model_impl._impl = model_impl._class_map[type_of_impl]()
    res_impl = model_impl._impl.fit_predict(dataset, k=2)
    assert isDataFrameEqual(res_client, res_impl)


"""
@pytest.mark.spark
@pytest.mark.parametrize("recommender", [PopRec], ids=["pop_rec_spark"])
def test_equal_preds_after_save_load_model_spark(recommender, tmp_path, request):
    path = (tmp_path / "test_spark").resolve()
    log = request.getfixturevalue("long_log_with_features")
    dataset = get_dataset_any_type(log)
    model = recommender()
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    assert isDataFrameEqual(new_pred, base_pred)


@pytest.mark.spark
@pytest.mark.parametrize("recommender", [PopRec], ids=["pop_rec_spark"])
def test_equal_preds_after_save_load_model_pandas(recommender, tmp_path, request):
    path = (tmp_path / "test_pandas").resolve()
    log = request.getfixturevalue(("long_log_with_features_pandas"))
    dataset = get_dataset_any_type(log)
    model = recommender()
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    assert isDataFrameEqual(new_pred, base_pred)


@pytest.mark.spark  # TODO: Save.load not working on pandas and polars. Check replay.utils.save 'todo' for context
@pytest.mark.parametrize("recommender", [PopRec], ids=["pop_rec_spark"])
def test_equal_preds_after_save_load_model_polars(recommender, tmp_path, request):
    path = (tmp_path / "test_polars").resolve()
    log = request.getfixturevalue(("long_log_with_features_polars"))
    dataset = get_dataset_any_type(log)
    model = recommender()
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    assert isDataFrameEqual(new_pred, base_pred)


@pytest.mark.spark
@pytest.mark.parametrize(
    "recommender, type_of_impl",
    [(PopRec, "spark"), (PopRec, "pandas"), (PopRec, "polars")],
    ids=["pop_rec_spark", "pop_rec_pandas", "pop_rec_polars"],
)
def test_equal_attributes_after_save_load_model(recommender, type_of_impl, tmp_path, request):
    from itertools import chain

    path = (tmp_path / "test_attributes").resolve()
    log = request.getfixturevalue(
        "long_log_with_features" if type_of_impl == "spark" else "long_log_with_features_" + type_of_impl
    )
    dataset = get_dataset_any_type(log)
    model = recommender()
    model.fit(dataset)
    save(model, path)
    loaded_model = load(path)
    all_attributes = chain(
        model.attributes_after_fit,
        [
            "_init_when_first_impl_arrived_args",
            "is_fitted",
            "is_spark",
            "is_polars",
            "is_pandas",
            "items_count",
            "queries_count",
        ],
    )
    for attr in all_attributes:
        if not isinstance(getattr(model, attr), tuple(DataFrameLike.__args__)) and not isinstance(
            getattr(model, attr), pd.Series
        ):
            assert getattr(model, attr) == getattr(loaded_model, attr)
        else:
            assert isDataFrameEqual(getattr(model, attr), getattr(loaded_model, attr))
"""
