# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models import (
    ALSWrap,
    ClusterRec,
    ItemKNN,
    PopRec,
    SLIM,
    UCB,
    Wilson,
    Word2VecRec,
    AssociationRulesItemRec,
)
from replay.models.base_rec import HybridRecommender, QueryRecommender

from tests.utils import (
    create_dataset,
    spark,
    log,
    log_to_pred,
    long_log_with_features,
    user_features,
    sparkDataFrameEqual,
)

SEED = 123


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(seed=SEED),
        ItemKNN(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
        AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_idx"),
    ],
    ids=[
        "als",
        "knn",
        "slim",
        "word2vec",
        "association_rules",
    ],
)
def test_predict_pairs_warm_items_only(log, log_to_pred, model):
    train_dataset = create_dataset(log)
    pred_dataset = create_dataset(log.unionByName(log_to_pred))
    model.fit(train_dataset)
    recs = model.predict(
        pred_dataset,
        k=3,
        queries=log_to_pred.select("user_idx").distinct(),
        items=log_to_pred.select("item_idx").distinct(),
        filter_seen_items=False,
    )

    pairs_pred = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        dataset=pred_dataset,
    )

    condition = ~sf.col("item_idx").isin([4, 5])
    if not model.can_predict_cold_queries:
        condition = condition & (sf.col("user_idx") != 4)

    sparkDataFrameEqual(
        pairs_pred.select("user_idx", "item_idx"),
        log_to_pred.filter(condition).select("user_idx", "item_idx"),
    )

    recs_joined = (
        pairs_pred.withColumnRenamed("relevance", "pairs_relevance")
        .join(recs, on=["user_idx", "item_idx"], how="left")
        .sort("user_idx", "item_idx")
    )

    assert np.allclose(
        recs_joined.select("relevance").toPandas().to_numpy(),
        recs_joined.select("pairs_relevance").toPandas().to_numpy(),
    )


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(seed=SEED),
        ItemKNN(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
        AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_idx"),
        PopRec(),
    ],
    ids=[
        "als",
        "knn",
        "slim",
        "word2vec",
        "association_rules",
        "pop_rec",
    ],
)
def test_predict_pairs_k(log, model):
    train_dataset = create_dataset(log)
    model.fit(train_dataset)

    pairs_pred_k = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        dataset=train_dataset,
        k=1,
    )

    pairs_pred = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        dataset=train_dataset,
        k=None,
    )

    assert (
        pairs_pred_k.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        == 0
    )

    assert (
        pairs_pred.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        > 0
    )


# for NeighbourRec and ItemVectorModel
@pytest.mark.parametrize(
    "model, metric",
    [
        (ALSWrap(seed=SEED), "euclidean_distance_sim"),
        (ALSWrap(seed=SEED), "dot_product"),
        (ALSWrap(seed=SEED), "cosine_similarity"),
        (Word2VecRec(seed=SEED, min_count=0), "cosine_similarity"),
        (ItemKNN(), None),
        (SLIM(seed=SEED), None),
        (AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_idx"), "lift"),
        (
            AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_idx"),
            "confidence",
        ),
        (
            AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_idx"),
            "confidence_gain",
        ),
    ],
    ids=[
        "als_euclidean",
        "als_dot",
        "als_cosine",
        "w2v_cosine",
        "knn",
        "slim",
        "association_rules_lift",
        "association_rules_confidence",
        "association_rules_confidence_gain",
    ],
)
def test_get_nearest_items(log, model, metric):
    train_dataset = create_dataset(log.filter(sf.col("item_idx") != 3))
    model.fit(train_dataset)
    res = model.get_nearest_items(items=[0, 1], k=2, metric=metric)

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_idx"].values()) == {
        0,
        1,
    }

    res = model.get_nearest_items(items=[0, 1], k=1, metric=metric)
    assert res.count() == 2

    # filter neighbours
    res = model.get_nearest_items(
        items=[0, 1],
        k=4,
        metric=metric,
        candidates=[0, 3],
    )
    assert res.count() == 1
    assert (
        len(
            set(res.toPandas().to_dict()["item_idx"].values()).difference(
                {0, 1}
            )
        )
        == 0
    )


def test_filter_seen(log):
    model = PopRec()
    # filter seen works with empty log to filter (cold_user)
    train_dataset = create_dataset(log.filter(sf.col("user_idx") != 0))
    pred_dataset = create_dataset(log)
    model.fit(train_dataset)
    pred = model.predict(dataset=pred_dataset, queries=[3], k=5)
    assert pred.count() == 2

    # filter seen works with log not presented during training (for user1)
    pred = model.predict(dataset=pred_dataset, queries=[0], k=5)
    assert pred.count() == 1

    # filter seen turns off
    pred = model.predict(dataset=pred_dataset, queries=[0], k=5, filter_seen_items=False)
    assert pred.count() == 4


def fit_predict_selected(model, train_log, inf_log, user_features, queries):
    train_dataset = create_dataset(train_log, user_features=user_features)
    pred_dataset = create_dataset(inf_log, user_features=user_features)
    model.fit(train_dataset)
    return model.predict(dataset=pred_dataset, queries=queries, k=1)


@pytest.mark.parametrize(
    "model",
    [
        ClusterRec(num_clusters=2),
        ItemKNN(),
        SLIM(seed=SEED),
        PopRec(),
        Word2VecRec(seed=SEED, min_count=0),
        AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_idx"),
    ],
    ids=[
        "cluster",
        "knn",
        "slim",
        "pop_rec",
        "word2vec",
        "association_rules",
    ],
)
def test_predict_new_queries(model, long_log_with_features, user_features):
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=user_features.drop("gender"),
        queries=[0],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == 0


@pytest.mark.parametrize(
    "model",
    [
        ClusterRec(num_clusters=2),
        PopRec(),
    ],
    ids=[
        "cluster",
        "pop_rec",
    ],
)
def test_predict_cold_queries(model, long_log_with_features, user_features):
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        user_features=user_features.drop("gender"),
        queries=[0],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == 0


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(rank=2, seed=SEED),
        ItemKNN(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
        AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_idx"),
    ],
    ids=[
        "als",
        "knn",
        "slim",
        "word2vec",
        "association_rules",
    ],
)
def test_predict_cold_and_new_filter_out(model, long_log_with_features):
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=None,
        queries=[0, 3],
    )
    # assert new/cold queries are filtered out in `predict`
    if not model.can_predict_cold_queries:
        assert pred.count() == 0
    else:
        assert 1 <= pred.count() <= 2


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(rank=2, seed=SEED),
        PopRec(),
        ItemKNN(),
    ],
    ids=[
        "als",
        "pop_rec",
        "knn",
    ],
)
def test_predict_pairs_to_file(spark, model, long_log_with_features, tmp_path):
    train_dataset = create_dataset(long_log_with_features)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit(train_dataset)
    model.predict_pairs(
        dataset=train_dataset,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select(
            "user_idx", "item_idx"
        ),
        recs_file_path=path,
    )
    pred_cached = model.predict_pairs(
        dataset=train_dataset,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select(
            "user_idx", "item_idx"
        ),
        recs_file_path=None,
    )
    pred_from_file = spark.read.parquet(path)
    sparkDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(rank=2, seed=SEED),
        PopRec(),
        ItemKNN(),
    ],
    ids=[
        "als",
        "pop_rec",
        "knn",
    ],
)
def test_predict_to_file(spark, model, long_log_with_features, tmp_path):
    train_dataset = create_dataset(long_log_with_features)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit_predict(train_dataset, k=10, recs_file_path=path)
    pred_cached = model.predict(
        train_dataset, k=10, recs_file_path=None
    )
    pred_from_file = spark.read.parquet(path)
    sparkDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.xfail
@pytest.mark.parametrize("add_cold_items", [True, False])
@pytest.mark.parametrize("predict_cold_only", [True, False])
@pytest.mark.parametrize(
    "model",
    [
        PopRec(),
        Wilson(sample=True),
        Wilson(sample=False),
        UCB(sample=True),
        UCB(sample=False),
    ],
    ids=[
        "pop_rec",
        "wilson_sample",
        "wilson",
        "UCB_sample",
        "UCB",
    ],
)
def test_add_cold_items_for_nonpersonalized(
    model, add_cold_items, predict_cold_only, long_log_with_features
):
    num_warm = 5
    # k is greater than the number of warm items to check if
    # the cold items are presented in prediction
    k = 6
    log = (
        long_log_with_features
        if not isinstance(model, (Wilson, UCB))
        else long_log_with_features.withColumn(
            "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
        )
    )
    train_log = log.filter(sf.col("item_idx") < num_warm)
    train_dataset = create_dataset(train_log)
    model.fit(train_dataset)
    # ucb always adds cold items to prediction
    if not isinstance(model, UCB):
        model.add_cold_items = add_cold_items

    items = log.select("item_idx").distinct()
    if predict_cold_only:
        items = items.filter(sf.col("item_idx") >= num_warm)

    pred_dataset = create_dataset(log.filter(sf.col("item_idx") < num_warm))
    pred = model.predict(
        dataset=pred_dataset,
        queries=[1],
        items=items,
        k=k,
        filter_seen_items=False,
    )

    if isinstance(model, UCB) or add_cold_items:
        assert pred.count() == min(k, items.count())
        if predict_cold_only:
            assert pred.select(sf.min("item_idx")).collect()[0][0] >= num_warm
            assert pred.select("relevance").distinct().count() == 1
    else:
        if predict_cold_only:
            assert pred.count() == 0
        else:
            # ucb always adds cold items to prediction
            assert pred.select(sf.max("item_idx")).collect()[0][0] < num_warm
            assert pred.count() == min(
                k,
                train_log.select("item_idx")
                .distinct()
                .join(items, on="item_idx")
                .count(),
            )


@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(),
        SLIM(seed=SEED),
    ],
    ids=[
        "knn",
        "slim",
    ],
)
def test_similarity_metric_raises(log, model):
    with pytest.raises(
        ValueError,
        match="This class does not support changing similarity metrics",
    ):
        train_dataset = create_dataset(log)
        model.fit(train_dataset)
        model.similarity_metric = "some"
