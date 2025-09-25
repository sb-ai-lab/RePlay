import logging

import pytest

from replay.utils import FeatureUnavailableError
from tests.utils import (
    create_dataset,
    sparkDataFrameEqual,
    sparkDataFrameNotEqual,
)


@pytest.fixture(scope="module")
def model():
    from replay.models import ALSWrap

    return ALSWrap()


@pytest.mark.conditional
@pytest.mark.parametrize("borders", [{"beta": [1, 2]}, {"lambda_": [1, 2]}])
def test_partial_borders(borders):
    from replay.models import SLIM

    model = SLIM()
    res = model._prepare_param_borders(borders)
    assert len(res) == len(model._search_space)


@pytest.mark.conditional
def test_ItemKNN(log):
    from replay.models import ItemKNN

    model = ItemKNN()
    dataset = create_dataset(log)
    res = model.optimize(dataset, dataset, k=2, budget=1)
    assert isinstance(res["num_neighbours"], int)


@pytest.mark.core
def test_ItemKNN_with_stub():
    from replay.models import ItemKNN

    model = ItemKNN()
    with pytest.raises(FeatureUnavailableError):
        _ = model.optimize(None, None, k=2, budget=1)


@pytest.mark.conditional
@pytest.mark.parametrize(
    "borders",
    [
        {"wrong_name": None},
        {"rank": None},
        {"rank": 2},
        {"rank": [1]},
        {"rank": [1, 2, 3]},
    ],
    ids=[
        "wrong name",
        "None border",
        "int border",
        "border's too short",
        "border's too long",
    ],
)
def test_bad_borders(model, borders):
    with pytest.raises(ValueError):
        model._prepare_param_borders(borders)


@pytest.mark.conditional
@pytest.mark.parametrize("borders", [None, {"rank": [5, 9]}])
def test_correct_borders(model, borders):
    res = model._prepare_param_borders(borders)
    assert res.keys() == model._search_space.keys()
    assert "rank" in res
    assert isinstance(res["rank"], dict)
    assert res["rank"].keys() == model._search_space["rank"].keys()


@pytest.mark.conditional
@pytest.mark.parametrize("borders,answer", [(None, True), ({"rank": [-10, -1]}, False)])
def test_param_in_borders(model, borders, answer):
    search_space = model._prepare_param_borders(borders)
    assert model._init_params_in_search_space(search_space) == answer


@pytest.mark.conditional
def test_missing_categorical_borders():
    from replay.models import ItemKNN

    borders = {"num_neighbours": [5, 10]}
    model = ItemKNN(weighting="bm25")
    search_space = model._prepare_param_borders(borders)
    assert search_space["weighting"]["args"] == ["bm25"]


@pytest.mark.conditional
def test_it_works(model, log):
    dataset = create_dataset(log)
    assert model._params_tried() is False
    res = model.optimize(dataset, dataset, k=2, budget=1)
    assert isinstance(res["rank"], int)


@pytest.mark.conditional
def test_empty_search_space(log, caplog):
    from replay.models import ItemKNN

    with caplog.at_level(logging.WARNING):
        model = ItemKNN()
        model._search_space = None
        dataset = create_dataset(log)
        res = model.optimize(dataset, dataset, k=2, budget=1)
        assert f"{model} has no hyper parameters to optimize" in caplog.text
        assert res is None


@pytest.mark.conditional
def test_filter_dataset_features(log, all_users_features, item_features):
    from replay.models import ItemKNN

    model = ItemKNN()
    dataset = create_dataset(log, all_users_features, item_features)
    filtered_dataset = model._filter_dataset_features(dataset)
    sparkDataFrameEqual(dataset.interactions, filtered_dataset.interactions)
    sparkDataFrameEqual(dataset.query_features, filtered_dataset.query_features)
    sparkDataFrameNotEqual(dataset.item_features, filtered_dataset.item_features)
