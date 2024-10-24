import logging
from datetime import datetime

import pytest

from replay.data import get_schema
from replay.models import LinUCB
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import create_dataset, sparkDataFrameEqual

if PYSPARK_AVAILABLE:
    from pyspark.ml.feature import StringIndexer
    from pyspark.sql import functions as sf
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

    INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


@pytest.fixture(scope="module")
def log(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 8, 22), 4.0],
            [0, 2, datetime(2019, 8, 23), 3.0],
            [0, 1, datetime(2019, 8, 27), 2.0],
            [1, 3, datetime(2019, 8, 24), 3.0],
            [1, 0, datetime(2019, 8, 25), 4.0],
            [2, 1, datetime(2019, 8, 26), 5.0],
            [2, 0, datetime(2019, 8, 26), 5.0],
            [2, 2, datetime(2019, 8, 26), 3.0],
            [3, 1, datetime(2019, 8, 26), 5.0],
            [3, 0, datetime(2019, 8, 26), 5.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture(scope="module")
def empty_log(spark):
    return spark.createDataFrame(
        data=[],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture(scope="module")
def user_features(spark):
    USER_FEATURES_SCHEMA = StructType(
        [
            StructField("user_idx", IntegerType(), False),
            StructField("age", DoubleType(), True),
            StructField("mood", DoubleType(), True),
            StructField("gender", StringType(), True),
        ]
    )
    return spark.createDataFrame(
        [
            (0, 20.0, -3.0, "M"),
            (1, 30.0, 4.0, "F"),
            (2, 75.0, -1.0, "M"),
            (3, 35.0, 42.0, "M"),
        ],
        schema=USER_FEATURES_SCHEMA,
    )


@pytest.fixture(scope="module")
def item_features(spark):
    ITEM_FEATURES_SCHEMA = StructType(
        [
            StructField("item_idx", IntegerType(), False),
            StructField("iq", DoubleType(), True),
            StructField("class", StringType(), True),
            StructField("color", StringType(), True),
        ]
    )
    return spark.createDataFrame(
        [
            (0, 4.0, "cat", "black"),
            (1, 10.0, "dog", "green"),
            (2, 7.0, "mouse", "yellow"),
            (3, -1.0, "cat", "yellow"),
            (4, 11.0, "dog", "white"),
            (5, 0.0, "mouse", "yellow"),
        ],
        schema=ITEM_FEATURES_SCHEMA,
    )


@pytest.fixture(scope="module")
def log_linucb(log):
    return log.withColumn("relevance", sf.when(sf.col("relevance") > 3, 1).otherwise(0))


@pytest.fixture(scope="module")
def user_features_linucb(user_features):
    indexer = StringIndexer(inputCol="gender", outputCol="gender_Indexed")
    indexerModel = indexer.fit(user_features)
    indexed_df = indexerModel.transform(user_features)
    indexed_df = indexed_df.drop("gender")
    return indexed_df


@pytest.fixture(scope="module")
def item_features_linucb(item_features):
    indexer = StringIndexer(inputCol="class", outputCol="class_Indexed")
    indexerModel = indexer.fit(item_features)
    indexed_df = indexerModel.transform(item_features)
    indexed_df = indexed_df.drop("class")

    indexer = StringIndexer(inputCol="color", outputCol="color_Indexed")
    indexerModel = indexer.fit(indexed_df)
    indexed_df = indexerModel.transform(indexed_df)
    indexed_df = indexed_df.drop("color")
    return indexed_df


@pytest.fixture(params=[LinUCB(eps=-10.0, alpha=1.0, is_hybrid=False)], scope="module")
def fitted_model_disjoint(request, log_linucb, user_features_linucb, item_features_linucb):
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)
    model = request.param
    model.fit(dataset)
    return model


@pytest.fixture(params=[LinUCB(eps=-10.0, alpha=1.0, is_hybrid=True)], scope="module")
def fitted_model_hybrid(request, log_linucb, user_features_linucb, item_features_linucb):
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)
    model = request.param
    model.fit(dataset)
    return model


@pytest.mark.spark
def test_predict_disjoint(fitted_model_disjoint, log_linucb, user_features_linucb, item_features_linucb):
    # fixed seed provides reproducibility (the same prediction every time),
    # non-fixed provides diversity (predictions differ every time)

    # add more items to get more randomness
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)
    pred = fitted_model_disjoint.predict(
        dataset, queries=user_features_linucb.select("user_idx"), items=item_features_linucb.select("item_idx"), k=1
    )
    pred_checkpoint = pred.localCheckpoint()
    pred.unpersist()

    # predictions are equal/non-equal after model re-fit
    fitted_model_disjoint.fit(dataset)

    pred_after_refit = fitted_model_disjoint.predict(
        dataset, queries=user_features_linucb.select("user_idx"), items=item_features_linucb.select("item_idx"), k=1
    )
    sparkDataFrameEqual(pred_checkpoint, pred_after_refit)

    # predictions are equal/non-equal when call `predict repeatedly`
    pred_after_refit_checkpoint = pred_after_refit.localCheckpoint()
    pred_after_refit.unpersist()
    pred_repeat = fitted_model_disjoint.predict(
        dataset, queries=user_features_linucb.select("user_idx"), items=item_features_linucb.select("item_idx"), k=1
    )
    sparkDataFrameEqual(pred_after_refit_checkpoint, pred_repeat)


@pytest.mark.spark
def test_predict_hybrid(fitted_model_hybrid, log_linucb, user_features_linucb, item_features_linucb):
    # fixed seed provides reproducibility (the same prediction every time),
    # non-fixed provides diversity (predictions differ every time)

    # add more items to get more randomness
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)
    pred = fitted_model_hybrid.predict(
        dataset, queries=user_features_linucb.select("user_idx"), items=item_features_linucb.select("item_idx"), k=1
    )
    pred_checkpoint = pred.localCheckpoint()
    pred.unpersist()

    # predictions are equal/non-equal after model re-fit
    fitted_model_hybrid.fit(dataset)

    pred_after_refit = fitted_model_hybrid.predict(
        dataset, queries=user_features_linucb.select("user_idx"), items=item_features_linucb.select("item_idx"), k=1
    )
    sparkDataFrameEqual(pred_checkpoint, pred_after_refit)

    # predictions are equal/non-equal when call `predict repeatedly`
    pred_after_refit_checkpoint = pred_after_refit.localCheckpoint()
    pred_after_refit.unpersist()
    pred_repeat = fitted_model_hybrid.predict(
        dataset, queries=user_features_linucb.select("user_idx"), items=item_features_linucb.select("item_idx"), k=1
    )
    sparkDataFrameEqual(pred_after_refit_checkpoint, pred_repeat)


@pytest.mark.spark
def test_predict_empty_log_disjoint(fitted_model_disjoint, user_features_linucb, item_features_linucb, empty_log):
    empty_dataset = create_dataset(empty_log, user_features_linucb, item_features_linucb)

    users = user_features_linucb.select("user_idx").distinct()
    pred_empty = fitted_model_disjoint.predict(empty_dataset, queries=users, k=1)
    assert pred_empty.count() == users.count()


@pytest.mark.spark
def test_predict_empty_log_hybrid(fitted_model_hybrid, user_features_linucb, item_features_linucb, empty_log):
    empty_dataset = create_dataset(empty_log, user_features_linucb, item_features_linucb)

    users = user_features_linucb.select("user_idx").distinct()
    pred_empty = fitted_model_hybrid.predict(empty_dataset, queries=users, k=1)
    assert pred_empty.count() == users.count()


@pytest.mark.spark
def test_optimize_disjoint(fitted_model_disjoint, log_linucb, user_features_linucb, item_features_linucb, caplog):
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)
    with caplog.at_level(logging.WARNING):
        fitted_model_disjoint.optimize(
            dataset,
            dataset,
            {"eps": [-20.0, 10.0], "alpha": [0.001, 10.0]},
            k=1,
            budget=1,
        )


@pytest.mark.spark
def test_optimize_hybrid(fitted_model_hybrid, log_linucb, user_features_linucb, item_features_linucb, caplog):
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)
    with caplog.at_level(logging.WARNING):
        fitted_model_hybrid.optimize(
            dataset,
            dataset,
            {"eps": [-20.0, 10.0], "alpha": [0.001, 10.0]},
            k=1,
            budget=1,
        )


@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
    ids=["k=1", "k=2", "k=3"],
)
@pytest.mark.spark
def test_predict_k_disjoint(fitted_model_disjoint, user_features_linucb, item_features_linucb, log_linucb, k):
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)

    users = user_features_linucb.select("user_idx").distinct()
    pred = fitted_model_disjoint.predict(dataset, queries=users, k=k)
    assert pred.count() == users.count() * k


@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
    ids=["k=1", "k=2", "k=3"],
)
@pytest.mark.spark
def test_predict_k_hybrid(fitted_model_hybrid, user_features_linucb, item_features_linucb, log_linucb, k):
    dataset = create_dataset(log_linucb, user_features_linucb, item_features_linucb)

    users = user_features_linucb.select("user_idx").distinct()
    pred = fitted_model_hybrid.predict(dataset, queries=users, k=k)
    assert pred.count() == users.count() * k
