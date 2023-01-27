# pylint: disable-all
import pytest
from pyspark.sql import DataFrame

from replay.constants import LOG_SCHEMA
from replay.models import CQL
from tests.utils import spark, log


@pytest.fixture
def model():
    model = CQL(top_k=1, n_epochs=3)
    return model


def test_works(log: DataFrame, model: CQL):
    model.fit(log)
    recs = model.predict(log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 0
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0
    assert recs.loc[recs["user_idx"] == 2, "item_idx"].iloc[0] == 0
    assert recs.loc[recs["user_idx"] == 3, "item_idx"].iloc[0] == 0
