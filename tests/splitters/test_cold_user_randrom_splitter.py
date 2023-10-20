# pylint: disable-all
from datetime import datetime

import numpy as np
import pytest

from replay.splitters import ColdUserRandomSplitter


@pytest.fixture
def log():
    import pandas as pd

    return pd.DataFrame(
        {
            "user_idx": list(range(5000)),
            "item_idx": list(range(5000)),
            "relevance": [1] * 5000,
        }
    )


def test(log):
    ratio = 0.25
    cold_user_splitter = ColdUserRandomSplitter(ratio)
    cold_user_splitter.seed = 27
    train, test = cold_user_splitter.split(log)
    test_users = test.toPandas().user_idx.unique()
    train_users = train.toPandas().user_idx.unique()
    assert not np.isin(test_users, train_users).any()
    real_ratio = len(test_users) / len(log)
    assert np.isclose(real_ratio, ratio, atol=0.01)  # Spark weights are random ¯\_(ツ)_/¯
