# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from pyspark.sql import functions as sf

from replay.models import AssociationRulesItemRec
from tests.utils import log, spark, sparkDataFrameEqual


@pytest.fixture
def model():
    model = AssociationRulesItemRec(min_item_count=1, min_pair_count=1)
    return model


def test_predict_raises(log, model):
    model.fit(log)
    with pytest.raises(
        NotImplementedError,
        match=r"item-to-user predict is not implemented for AssociationRulesItemRec,.*",
    ):
        model.predict(log, 1)


def test_invalid_metric_raises(log, model):
    model.fit(log)
    with pytest.raises(
        ValueError,
        match=r"Select one of the valid distance metrics: \['lift', 'confidence_gain'\]",
    ):
        model.get_nearest_items(log.select("item_id"), k=1, metric="invalid")


def test_works(log, model):
    model.fit(log)
    assert hasattr(model, "pair_metrics")
    model.pair_metrics.count()


def test_calculation(model, log):
    model.fit(log)

    # convert ids back
    pairs_metrics = model.get_pair_metrics()

    # recalculate for item_3 as antecedent and item_2 as consequent
    test_row = pairs_metrics.filter(
        (sf.col("antecedent") == "item3") & (sf.col("consequent") == "item2")
    ).toPandas()

    # calculation
    num_of_sessions = log.select("user_id").distinct().count()
    count_3, count_2, count_3_2 = 2, 3, 2
    confidence_3_2 = count_3_2 / count_3
    confidence_not_3_2 = (count_2 - count_3_2) / (num_of_sessions - count_3)

    assert test_row["confidence"][0] == confidence_3_2
    assert test_row["lift"][0] == confidence_3_2 / (count_2 / num_of_sessions)
    assert (
        test_row["confidence_gain"][0] == confidence_3_2 / confidence_not_3_2
    )


def test_get_nearest_items(log, model):
    model.fit(log)
    res = model.get_nearest_items(
        items=["item3"],
        k=10,
        metric="confidence_gain",
        candidates=["item2", "item4"],
    )

    assert res.count() == 1
    assert res.select("neighbour_item_id").collect()[0][0] == "item2"
    assert res.select("confidence_gain").collect()[0][0] == 2.0
    assert res.select("lift").collect()[0][0] == 4 / 3

    res = model.get_nearest_items(
        items=["item3"],
        k=10,
        metric="lift",
    )
    assert res.count() == 2

    model._clear_cache()
