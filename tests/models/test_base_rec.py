# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, pointless-statement
from typing import Optional

import pytest
from pandas import DataFrame

from replay.models import Recommender
from tests.utils import log, spark


# pylint: disable=missing-class-docstring, too-many-arguments
class DerivedRec(Recommender):
    @property
    def _init_args(self):
        return {}

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
def model():
    return DerivedRec()


def test_users_count(model, log):
    with pytest.raises(AttributeError):
        model._user_dim
    model.fit(log)
    assert model._user_dim == 4


def test_items_count(model, log):
    with pytest.raises(AttributeError):
        model._item_dim
    model.fit(log)
    assert model._item_dim == 4


def test_str(model):
    assert str(model) == "DerivedRec"
