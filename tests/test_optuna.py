# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from replay.models import ALSWrap, RandomRec
from tests.utils import log, spark


@pytest.mark.parametrize(
    "model,search_space,res_params",
    [
        (ALSWrap(rank=2), {"rank": [1, 10]}, {"rank": 2}),
        (RandomRec(), None, {"distribution": "uniform", "alpha": 0.0}),
    ],
    ids=[
        "int",
        "cat_and_float",
    ],
)
def test_param_types(log, model, search_space, res_params):
    res = model.optimize(log, log, k=2, budget=1, param_grid=search_space)
    for param, value in res_params.items():
        assert getattr(model, param) == value == res[param]


@pytest.mark.parametrize(
    "model_params,search_space",
    [({"rank": 2}, None), ({}, {"rank": [2, 5]})],
    ids=[
        "less_default_space",
        "greater_defined_space",
    ],
)
def test_init_params_outside(log, model_params, search_space):
    model = ALSWrap(**model_params)
    init_rank = model.rank
    res = model.optimize(log, log, k=2, param_grid=search_space, budget=1)
    assert res["rank"] == model.rank
    assert res["rank"] != init_rank


@pytest.mark.parametrize(
    "model_params,search_space",
    [({"rank": 20}, None), ({"rank": 3}, {"rank": [2, 5]})],
    ids=[
        "default_space",
        "defined_space",
    ],
)
def test_init_params_inside(log, model_params, search_space):
    model = ALSWrap(**model_params)
    init_rank = model.rank
    res = model.optimize(log, log, k=2, param_grid=search_space, budget=1)
    assert res["rank"] == model.rank
    assert res["rank"] == init_rank
