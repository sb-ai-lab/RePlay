# pylint: disable-all

import pandas as pd

from replay.models import ColdUser


train = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [1, 2, 3]})

user_features = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 5],
        "age": [18, 20, 80, 16, 69],
        "sex": [1, 0, 0, 1, 0],
    }
)

test = pd.DataFrame({"user_id": [4, 5], "item_id": [1, 2]})


def test_works():
    model = ColdUser()
    res = model.optimize(train, test, user_features, k=1, budget=1)
    assert type(res["n"]) == int
