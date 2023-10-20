# pylint: disable-all
import pandas as pd
import pytest

from replay.splitters.user_log_splitter import k_folds


@pytest.fixture
def df():
    df = pd.DataFrame({"user_idx": [1, 1, 1, 2, 2], "item_idx": [1, 2, 3, 4, 5]})
    return df


def test_sum(df):
    res = pd.DataFrame()
    for _, test in k_folds(df, n_folds=2, seed=1337):
        res = res.append(test.toPandas(), ignore_index=True)
    res = res.sort_values(["user_idx", "item_idx"]).reset_index(drop=True)
    assert all(res == df)


def test_wrong_type(df):
    with pytest.raises(ValueError):
        next(k_folds(df, splitter="totally not user"))
