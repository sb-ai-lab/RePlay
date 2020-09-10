# type: ignore
# pylint: disable-all

import pandas as pd
import pytest

from replay.models import KNN, Stack


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 1, 2, 1, 2, 3, 3, 3, 4, 4, 4],
            "item_id": [1, 2, 2, 3, 3, 4, 5, 7, 7, 5, 3, 4, 2, 5],
            "relevance": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    return df


def test_finishes(df):
    stack = Stack([KNN()], n_folds=2, budget=1)
    stack.fit_predict(df, 1)
    assert True
