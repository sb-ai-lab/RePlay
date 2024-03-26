import logging

import numpy as np
import pandas as pd
import pytest

from replay.data.dataset import Dataset
from replay.models import RandomRec
from replay.models.base_rec import HybridRecommender, NonPersonalizedRecommender, Recommender
from replay.utils import SparkDataFrame
from replay.utils.spark_utils import convert2spark
from tests.utils import (
    create_dataset,
)


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
        dataset: Dataset,
        k: int,
        queries: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        return dataset.interactions.select("user_idx", "item_idx", "relevance")


class DummyHybridRec(HybridRecommender):
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
        dataset: Dataset,
        k: int,
        queries: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        return dataset.interactions.select("user_idx", "item_idx", "relevance")


class NonPersRec(NonPersonalizedRecommender):
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
        dataset: Dataset,
        k: int,
        queries: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        return dataset.interactions.select("user_idx", "item_idx", "relevance")


@pytest.fixture
def model():
    return DerivedRec()


@pytest.fixture
def hybrid_model():
    return DummyHybridRec()


@pytest.fixture
def dataset(log, all_users_features, item_features):
    return create_dataset(log, all_users_features, item_features)


@pytest.mark.spark
def test_users_count(model, log):
    with pytest.raises(AttributeError):
        model._query_dim
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

    pred = model._predict_proba(dataset, K, users, items, filter_seen_items=False)

    assert pred.shape == (n_users, n_actions, K)
    assert np.allclose(pred.sum(1), np.ones(shape=(n_users, K)))


@pytest.mark.spark
def test_get_fit_counts(model, fake_fit_items, fake_fit_queries):
    assert not hasattr(model, "_num_items")
    assert not hasattr(model, "_num_queries")
    model.fit_items = fake_fit_items
    model.fit_queries = fake_fit_queries
    assert model.items_count == 7
    assert model.queries_count == 6
    assert model._num_items == 7
    assert model._num_queries == 6


@pytest.mark.spark
def test_get_fit_dims(model, fake_fit_items, fake_fit_queries):
    assert not hasattr(model, "_item_dim_size")
    assert not hasattr(model, "_query_dim_size")
    model.item_column = "item_idx"
    model.query_column = "user_idx"
    model.fit_items = fake_fit_items
    model.fit_queries = fake_fit_queries
    assert model._item_dim == 6
    assert model._query_dim == 4
    assert model._item_dim_size == 6
    assert model._query_dim_size == 4


@pytest.mark.spark
def test_predict_pairs(model, log, caplog):
    caplog.set_level(logging.WARNING, logger="replay")
    dataset = create_dataset(log)
    model.fit(dataset)

    res = model.predict_pairs(pairs=log.select("user_idx", "item_idx"), dataset=dataset, k=1)

    assert res.select("user_idx").count() == 4
    assert (
        "native predict_pairs is not implemented for this model. "
        "Falling back to usual predict method and filtering the results." in caplog.text
    )


@pytest.mark.spark
def test_get_features_raise(model, log):
    model.query_column = "fake_users"
    model.item_column = "fake_items"
    with pytest.raises(ValueError):
        model.get_features(ids=log.select("item_idx"))


@pytest.mark.spark
def test_get_features_warning(model, caplog):
    with caplog.at_level(logging.INFO):
        model._get_features(None, None)
        assert (
            f"get_features method is not defined for the model {model}. Features will not be returned." in caplog.text
        )


@pytest.mark.core
def test_get_nearest_items_not_implemented(model):
    with pytest.raises(NotImplementedError, match="item-to-item prediction is not implemented for.*"):
        model._get_nearest_items(None, None, None)


@pytest.mark.spark
def test_hybrid_base_model(hybrid_model, dataset, log):
    hybrid_model.fit(dataset)
    hybrid_model.predict(dataset, k=1, queries=log.select("user_idx"), filter_seen_items=False)
    hybrid_model.fit_predict(dataset, k=1)
    hybrid_model.predict_pairs(log.select("user_idx", "item_idx"), dataset)
    hybrid_model.get_features(log.select("user_idx", "item_idx"), None)


@pytest.mark.core
def test_non_pers_init_raise():
    with pytest.raises(ValueError):
        NonPersRec(add_cold_items=True, cold_weight=-42.1)


@pytest.mark.spark
def test_non_pers_check_rating_raise(log):
    with pytest.raises(ValueError, match="Rating values in interactions must be 0 or 1"):
        model = NonPersRec(add_cold_items=True, cold_weight=0.5)
        dataset = create_dataset(log)
        model._check_rating(dataset)


@pytest.mark.spark
def test_non_pers_calc_max_hist_len(log):
    model = NonPersRec(add_cold_items=True, cold_weight=0.5)
    dataset = create_dataset(log)
    model.fit(dataset)
    max_len = model._calc_max_hist_len(dataset, log.select("user_idx"))
    assert max_len == 3
    max_len = model._calc_max_hist_len(dataset, log.select("user_idx").limit(0))
    assert max_len == 0
