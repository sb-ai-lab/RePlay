# pylint: disable-all
import pandas as pd

from replay.scenarios import ColdUser


train = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [1, 2, 3]})

users_train = pd.DataFrame(
    {"user_id": [1, 2, 3], "age": [18, 20, 80], "sex": [1, 0, 0]}
)

users_test = pd.DataFrame({"user_id": [4, 5], "age": [16, 69], "sex": [1, 0]})

test = pd.DataFrame({"user_id": [4, 5], "item_id": [1, 2]})


def test_works():
    model = ColdUser()
    res = model.optimize(train, test, users_train, users_test, k=1, budget=1)
    assert type(res) == int
