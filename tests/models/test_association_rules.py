# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from replay.models import AssociationRulesItemRec
from tests.utils import create_dataset, log, log_to_pred, spark, sparkDataFrameEqual, sparkDataFrameNotEqual

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import functions as sf


@pytest.fixture
def model(log):
    dataset = create_dataset(log)
    model = AssociationRulesItemRec(min_item_count=1, min_pair_count=1, session_column="user_idx")
    model.fit(dataset)
    return model


def test_invalid_metric_raises(log, model):
    with pytest.raises(
        ValueError,
        match=r"Select one of the valid distance metrics: \['lift', 'confidence', 'confidence_gain'\]",
    ):
        model.get_nearest_items(log.select("item_idx"), k=1, metric="invalid")


def test_works(model):
    assert hasattr(model, "similarity")
    model.similarity.count()


def check_formulas(count_ant, count_cons, pair_count, num_sessions, test_row):
    confidence_ant_con = pair_count / count_ant
    confidence_not_ant_con = (count_cons - pair_count) / (
        num_sessions - count_ant
    )
    assert test_row["confidence"][0] == confidence_ant_con
    assert test_row["lift"][0] == confidence_ant_con / (
        count_cons / num_sessions
    )
    assert (
        test_row["confidence_gain"][0]
        == confidence_ant_con / confidence_not_ant_con
    )


def test_calculation(model, log):
    pairs_metrics = model.get_similarity
    # recalculate for item_3 as antecedent and item_2 as consequent
    test_row = pairs_metrics.filter(
        (sf.col("antecedent") == 2) & (sf.col("consequent") == 1)
    ).toPandas()
    check_formulas(
        count_ant=2,
        count_cons=3,
        pair_count=2,
        num_sessions=log.select("user_idx").distinct().count(),
        test_row=test_row,
    )


def test_calculation_with_weights(model, log):
    model = AssociationRulesItemRec(
        min_item_count=1, min_pair_count=1, use_rating=True, session_column="user_idx",
    )
    dataset = create_dataset(log)
    model.fit(dataset)
    pairs_metrics = model.get_similarity
    # recalculate for item_3 as antecedent and item_2 as consequent using relevance values as weight
    test_row = pairs_metrics.filter(
        (sf.col("antecedent") == 2) & (sf.col("consequent") == 1)
    ).toPandas()
    check_formulas(
        count_ant=6,
        count_cons=12,
        pair_count=5,
        num_sessions=log.select("user_idx").distinct().count(),
        test_row=test_row,
    )


def test_get_nearest_items(model):
    res = model.get_nearest_items(
        items=[2], k=10, metric="confidence_gain", candidates=[1, 3],
    )

    assert res.count() == 1
    assert res.select("confidence_gain").collect()[0][0] == 2.0

    res = model.get_nearest_items(items=[2], k=10, metric="lift",)
    assert res.count() == 2

    model._clear_cache()


def test_metric(log, log_to_pred, model):
    dataset = create_dataset(log)
    model.fit(dataset)

    pred_dataset = create_dataset(log.unionByName(log_to_pred))
    p_pred_metr_from_init_conf = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        dataset=pred_dataset,
    )

    model.similarity_metric = "confidence"

    p_pred_metr_from_user_conf = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        dataset=pred_dataset,
    )

    sparkDataFrameEqual(
        p_pred_metr_from_init_conf,
        p_pred_metr_from_user_conf,
    )

    model.similarity_metric = "lift"

    p_pred_metr_from_user_lift = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        dataset=pred_dataset,
    )

    sparkDataFrameNotEqual(
        p_pred_metr_from_user_conf,
        p_pred_metr_from_user_lift
    )


def test_similarity_metric_raises(log, model):
    with pytest.raises(ValueError, match="Select one of the valid metrics for predict:.*"):
        dataset = create_dataset(log)
        model.fit(dataset)
        model.similarity_metric = "invalid"
