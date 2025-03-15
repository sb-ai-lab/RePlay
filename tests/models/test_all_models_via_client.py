import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.models import UCB, ClusterRec, ItemKNN, PopRec, RandomRec, Wilson, client_model_list
from tests.utils import SparkDataFrame, get_dataset_any_type

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf

SEED = 123


@pytest.fixture(scope="module")
def log_binary_rating(log):
    return log.withColumn("relevance", sf.when(sf.col("relevance") > 3, 1).otherwise(0))


@pytest.mark.spark
@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(),
        PopRec(),
    ],
    ids=["knn", "pop_rec"],
)
def test_predict_pairs_k_spark(model, request):
    log = request.getfixturevalue("log")
    ds = get_dataset_any_type(log)
    train_dataset = get_dataset_any_type(log)
    pairs = ds.interactions.select("user_idx", "item_idx")
    model.fit(train_dataset)
    pairs_pred_k = model.predict_pairs(pairs=pairs, dataset=train_dataset, k=1)
    pairs_pred = model.predict_pairs(pairs=pairs, dataset=train_dataset, k=None)
    assert pairs_pred_k.groupBy("user_idx").count().filter(sf.col("count") > 1).count() == 0
    assert pairs_pred.groupBy("user_idx").count().filter(sf.col("count") > 1).count() > 0


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
def test_predict_pairs_k_core(dataset_type, model, request):
    if dataset_type == "pandas":
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
            counts_k = pairs_pred_k.groupby("user_idx").size()
            counts = pairs_pred.groupby("user_idx").size()
            assert all(counts_k <= 1)
            assert any(counts > 1)
    elif dataset_type == "polars":
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
            counts_k = pairs_pred_k.group_by("user_idx").count()
            counts = pairs_pred.group_by("user_idx").count()
            assert all(counts_k["count"].to_pandas().to_numpy() <= 1)
            assert any(counts["count"].to_pandas().to_numpy() > 1)
    else:
        msg = "Incorrect test"
        raise ValueError(msg)


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


def fit_predict_selected(model, train_log, inf_log, user_features, queries):
    train_dataset = get_dataset_any_type(train_log, user_features=user_features)
    pred_dataset = get_dataset_any_type(inf_log, user_features=user_features)
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
    pred_cached.toPandas().equals(pred_from_file)


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
            pred_cached.equals(pred_from_file)
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
            pred_cached.to_pandas().equals(pred_from_file)


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
    pred_cached.toPandas().equals(pred_from_file)


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
            pred_cached.equals(pred_from_file)
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
            pred_cached.to_pandas().equals(pred_from_file)


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
