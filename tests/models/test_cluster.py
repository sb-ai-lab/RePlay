# pylint: disable-all

import pandas as pd
import pytest

from replay.models import ClusterRec


train = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [1, 2, 3]})

user_features = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 5],
        "age": [18, 20, 80, 16, 69],
        "sex": [1, 0, 0, 1, 0],
    }
)

test = pd.DataFrame({"user_id": [4, 5], "item_id": [1, 2]})


@pytest.fixture
def model():
    return ClusterRec()


def test_works(model):
    model.fit(train, user_features)
    model.predict(user_features, k=1)
    res = model.optimize(train, test, user_features, k=1, budget=1)
    assert type(res["num_clusters"]) == int


def test_base_predict_pairs(model):
    model.fit(train, user_features=user_features)
    # calls predict_pairs from BaseRecommender
    res = model.predict_pairs(
        train.iloc[0:1, :], log=train, user_features=user_features
    )
    assert res.count() == 1
    assert res.select("item_id").collect()[0][0] == 1
