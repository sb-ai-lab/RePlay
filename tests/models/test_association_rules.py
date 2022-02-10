# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from pyspark.sql import functions as sf

from replay.models import AssociationRulesItemRec
from tests.utils import log, spark, sparkDataFrameEqual


@pytest.fixture
def model(log):
    model = AssociationRulesItemRec(min_item_count=1, min_pair_count=1)
    model.fit(log)
    return model


def test_predict_raises(log, model):
    with pytest.raises(
        NotImplementedError,
        match=r"item-to-user predict is not implemented for AssociationRulesItemRec,.*",
    ):
        model.predict(log, 1)


def test_invalid_metric_raises(log, model):
    with pytest.raises(
        ValueError,
        match=r"Select one of the valid distance metrics: \['lift', 'confidence_gain'\]",
    ):
        model.get_nearest_items(log.select("item_idx"), k=1, metric="invalid")


def test_works(model):
    assert hasattr(model, "pair_metrics")
    model.pair_metrics.count()


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
    pairs_metrics = model.get_pair_metrics
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
        min_item_count=1, min_pair_count=1, use_relevance=True
    )
    model.fit(log)
    pairs_metrics = model.get_pair_metrics
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
    assert res.select("neighbour_item_idx").collect()[0][0] == 1
    assert res.select("confidence_gain").collect()[0][0] == 2.0
    assert res.select("lift").collect()[0][0] == 4 / 3

    res = model.get_nearest_items(items=[2], k=10, metric="lift",)
    assert res.count() == 2

    model._clear_cache()
