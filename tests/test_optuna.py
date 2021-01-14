# pylint: disable-all
import pytest
import pandas as pd

from replay.models import ALSWrap


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [1, 2, 3, 4, 5],
            "relevance": [1, 1, 1, 1, 1],
        }
    )
    return df


def test_it_works(df):
    model = ALSWrap()
    res = model.optimize(df, df, k=2, budget=1)
    assert type(res["rank"]) == int
