# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, pointless-statement
import pytest

import numpy as np
import pandas as pd

from replay.data.dataset import Dataset
from replay.models import RandomRec, Recommender

from replay.utils import PandasDataFrame
from replay.utils.spark_utils import convert2spark
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


@pytest.mark.spark
def test_users_count(model, log):
    with pytest.raises(AttributeError):
        model._qiery_dim
    dataset = create_dataset(log)
    model.fit(dataset)
    assert model._query_dim == 4


@pytest.mark.spark
def test_items_count(model, log):
    with pytest.raises(AttributeError):
        model._item_dim
    dataset = create_dataset(log)
    model.fit(dataset)
    assert model._item_dim == 4


@pytest.mark.core
def test_str(model):
    assert str(model) == "DerivedRec"


@pytest.mark.spark
@pytest.mark.parametrize("sample", [True, False])
def test_predict_proba(log, sample, n_users=2, n_actions=5, K=3):
    users = convert2spark(pd.DataFrame({"user_idx": np.arange(n_users)}))
    items = convert2spark(pd.DataFrame({"item_idx": np.arange(n_actions)}))

    model = RandomRec(seed=42)
    model.sample = sample
    dataset = create_dataset(log)
    model.fit(dataset)

    pred = model._predict_proba(
        dataset, K, users, items, filter_seen_items=False
    )

    assert pred.shape == (n_users, n_actions, K)
    assert np.allclose(pred.sum(1), np.ones(shape=(n_users, K)))
