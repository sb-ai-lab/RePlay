# pylint: disable-all

import pandas as pd
import pytest

from replay.models import KNN, Stack
from replay.session_handler import get_spark_session, State


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
    State(get_spark_session(1, 1))
    stack = Stack([KNN()], n_folds=2, budget=1, seed=1)
    pred = stack.fit_predict(df, 1).toPandas()
    pred = (
        pred.loc[:, ["user_id", "item_id"]]
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    res = pd.DataFrame({"user_id": [1, 2, 3, 4], "item_id": [4, 5, 2, 3]})
    pd.testing.assert_frame_equal(pred, res)
