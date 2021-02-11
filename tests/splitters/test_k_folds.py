# pylint: disable-all
import pytest
import pandas as pd

from replay.splitters.user_log_splitter import k_folds


@pytest.fixture
def df():
    df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "item_id": [1, 2, 3, 4, 5]})
    return df


def test_sum(df):
    res = pd.DataFrame()
    for _, test in k_folds(df, n_folds=2, seed=1337):
        res = res.append(test.toPandas(), ignore_index=True)
    res = res.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    assert all(res == df)
