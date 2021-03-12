# pylint: disable-all
from datetime import datetime
from typing import Optional

import pytest
import numpy as np
from pandas import DataFrame
from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import Word2VecRec, Recommender
from replay.utils import vector_dot
from tests.utils import spark


class DerivedRec(Recommender):
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        pass

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        pass


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[["1", "2", "3", "4"]],
        schema=["item_id", "user_id", "timestamp", "relevance"],
    )


@pytest.fixture
def model():
    return DerivedRec()


@pytest.mark.parametrize("array", [None, [1, 2, 2, 3]])
def test_extract_if_needed(spark, model, array):
    log = spark.createDataFrame(data=[[1], [2], [3]], schema=["test"])
    assert sorted(
        list(model._get_ids(array or log, "test").toPandas()["test"])
    ) == [1, 2, 3]


def test_users_count(model, log):
    with pytest.raises(AttributeError):
        model.users_count
    model.fit(log)
    assert model.users_count == 1


def test_items_count(model, log):
    with pytest.raises(AttributeError):
        model.items_count
    model.fit(log)
    assert model.items_count == 1


def test_str(model):
    assert str(model) == "DerivedRec"
