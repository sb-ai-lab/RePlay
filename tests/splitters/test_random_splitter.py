# pylint: disable-all
import pytest
import numpy as np

from replay.splitters import RandomSplitter


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


test_sizes = [0.0, 1.0, 0.5, 0.42, 0.95]


@pytest.mark.parametrize("test_size", test_sizes)
def test_nothing_is_lost(test_size, log):
    splitter = RandomSplitter(
        test_size=test_size,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=7777,
    )
    train, test = splitter.split(log)
    real_test_size = test.count() / len(log)
    assert train.count() + test.count() == len(log)
    assert np.isclose(real_test_size, test_size, atol=0.01)


@pytest.mark.parametrize("test_size", [-1.0, 2.0])
def test_bad_test_size(log, test_size):
    with pytest.raises(ValueError):
        RandomSplitter(test_size)
