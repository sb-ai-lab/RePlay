# pylint: disable-all
import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models import ALSWrap
from replay.scenarios.two_stages.two_stages_scenario import (
    get_first_level_model_features,
)
from tests.utils import log, spark


@pytest.fixture
def model():
    model = ALSWrap(2, implicit_prefs=False)
    model._seed = 42
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        assert pred.count() == 4
    except:  # noqa
        pytest.fail()


def test_diff_feedback_type(log, model):
    pred_exp = model.fit_predict(log, k=1)
    model.implicit_prefs = True
    pred_imp = model.fit_predict(log, k=1)
    assert not np.allclose(
        pred_exp.toPandas().sort_values("user_idx")["relevance"].values,
        pred_imp.toPandas().sort_values("user_idx")["relevance"].values,
    )


def test_enrich_with_features(log, model):
    model.fit(log.filter(sf.col("user_idx").isin([0, 2])))
    res = get_first_level_model_features(
        model, log.filter(sf.col("user_idx").isin([0, 1]))
    )

    cold_user_and_item = res.filter(
        (sf.col("user_idx") == 1) & (sf.col("item_idx") == 3)
    )
    row_dict = cold_user_and_item.collect()[0].asDict()
    assert row_dict["_if_0"] == row_dict["_uf_0"] == row_dict["_fm_1"] == 0.0

    warm_user_and_item = res.filter(
        (sf.col("user_idx") == 0) & (sf.col("item_idx") == 0)
    )
    row_dict = warm_user_and_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_fm_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [4.093189725967505, row_dict["_fm_1"]],
    )

    cold_user_warm_item = res.filter(
        (sf.col("user_idx") == 1) & (sf.col("item_idx") == 0)
    )
    row_dict = cold_user_warm_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_if_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [-2.938199281692505, 0],
    )
