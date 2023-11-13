# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, pointless-statement
from typing import Optional

import pytest
from pandas import DataFrame
from replay.data.dataset import Dataset

from replay.models import Recommender
from tests.utils import spark, log, create_dataset


# pylint: disable=missing-class-docstring, too-many-arguments
class DerivedRec(Recommender):
    @property
    def _init_args(self):
        return {}

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        pass

    def _predict(
        self,
        dataset: DataFrame,
        k: int,
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        pass


@pytest.fixture
def model():
    return DerivedRec()


def test_users_count(model, log):
    with pytest.raises(AttributeError):
        model._qiery_dim
    dataset = create_dataset(log)
    model.fit(dataset)
    assert model._query_dim == 4


def test_items_count(model, log):
    with pytest.raises(AttributeError):
        model._item_dim
    dataset = create_dataset(log)
    model.fit(dataset)
    assert model._item_dim == 4


def test_str(model):
    assert str(model) == "DerivedRec"
