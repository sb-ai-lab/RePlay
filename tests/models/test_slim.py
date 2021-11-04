# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
import numpy as np

from replay.models import SLIM
from tests.utils import log, spark


@pytest.fixture
def model():
    return SLIM(0.0, 0.01, seed=42)


def test_fit(log, model):
    model.fit(log)
    assert np.allclose(
        model.similarity.toPandas()
        .sort_values(["item_id_one", "item_id_two"])
        .to_numpy(),
        [
            (0, 1, 0.60048005),
            (0, 2, 0.12882786),
            (0, 3, 0.12860215),
            (1, 0, 1.06810235),
            (1, 2, 0.23784898),
            (2, 0, 0.25165837),
            (2, 1, 0.26372437),
            (3, 0, 1.32888889),
        ],
    )


def test_predict(log, model):
    model.fit(log)
    recs = model.predict(log, k=1)
    assert np.allclose(
        recs.toPandas()
        .sort_values(["user_id", "item_id"], ascending=False)
        .relevance,
        [0.4955047, 0.12860215, 0.60048005, 0.12860215],
    )


@pytest.mark.parametrize(
    "beta,lambda_", [(0.0, 0.0), (-0.1, 0.1), (0.1, -0.1)]
)
def test_exceptions(beta, lambda_):
    with pytest.raises(ValueError):
        SLIM(beta, lambda_)
