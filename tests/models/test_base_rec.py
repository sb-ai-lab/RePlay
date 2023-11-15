# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, pointless-statement
import pytest

from replay.data.dataset import Dataset
from replay.models import Recommender
from replay.utils import PandasDataFrame
from tests.utils import create_dataset, log, spark


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
        dataset: PandasDataFrame,
        k: int,
        queries: PandasDataFrame,
        items: PandasDataFrame,
        filter_seen_items: bool = True,
    ) -> PandasDataFrame:
        pass


@pytest.fixture
def model():
    return DerivedRec()


@pytest.mark.torch
def test_users_count(model, log):
    with pytest.raises(AttributeError):
        model._qiery_dim
    dataset = create_dataset(log)
    model.fit(dataset)
    assert model._query_dim == 4


@pytest.mark.torch
def test_items_count(model, log):
    with pytest.raises(AttributeError):
        model._item_dim
    dataset = create_dataset(log)
    model.fit(dataset)
    assert model._item_dim == 4


@pytest.mark.core
def test_str(model):
    assert str(model) == "DerivedRec"
