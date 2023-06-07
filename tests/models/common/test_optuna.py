# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from replay.models import (
    ADMMSLIM,
    ALSWrap,
    AssociationRulesItemRec,
    ClusterRec,
    ItemKNN,
    LightFMWrap,
    MultVAE,
    NeuroMF,
    PopRec,
    RandomRec,
    SLIM,
    Word2VecRec,
)
from tests.utils import log, spark


@pytest.fixture
def model():
    return ALSWrap()


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


@pytest.mark.parametrize("borders", [None, {"rank": [5, 9]}])
def test_correct_borders(model, borders):
    res = model._prepare_param_borders(borders)
    assert res.keys() == model._search_space.keys()
    assert "rank" in res
    assert isinstance(res["rank"], dict)
    assert res["rank"].keys() == model._search_space["rank"].keys()


@pytest.mark.parametrize("borders", [{"beta": [1, 2]}, {"lambda_": [1, 2]}])
def test_partial_borders(borders):
    model = SLIM()
    res = model._prepare_param_borders(borders)
    assert len(res) == len(model._search_space)


@pytest.mark.parametrize(
    "borders,answer", [(None, True), ({"rank": [-10, -1]}, False)]
)
def test_param_in_borders(model, borders, answer):
    search_space = model._prepare_param_borders(borders)
    assert model._init_params_in_search_space(search_space) == answer


@pytest.mark.parametrize(
    "model",
    [
        ADMMSLIM(),
        ALSWrap(),
        AssociationRulesItemRec(min_item_count=1, min_pair_count=0),
        ItemKNN(),
        MultVAE(epochs=1),
        NeuroMF(epochs=1),
        SLIM(),
        Word2VecRec(min_count=0),
    ],
    ids=[
        "admm_slim",
        "als",
        "association_rules",
        "knn",
        "multvae",
        "neuromf",
        "slim",
        "word2vec",
    ],
)
def test_works(model, log):
    assert model._params_tried() is False
    model.optimize(log, log, k=2, budget=1)
    assert model._params_tried() is True
    model.optimize(log, log, k=2, budget=1)
    assert len(model.study.trials) == 1
    model.optimize(log, log, k=2, budget=1, new_study=False)
    assert len(model.study.trials) == 2


def test_ItemKNN(log):
    model = ItemKNN()
    res = model.optimize(log, log, k=2, budget=1)
    assert isinstance(res["num_neighbours"], int)
