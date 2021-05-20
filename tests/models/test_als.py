# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models import ALSWrap
from tests.utils import log, spark


@pytest.fixture
def model():
    model = ALSWrap(1)
    model._seed = 42
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        np.allclose(
            pred.toPandas().sort_values("user_id")["relevance"].values,
            [0.559088, 0.816796, 0.566497, 0.783326],
        )
    except:  # noqa
        pytest.fail()


def test_predict_pairs(log, model):
    try:
        model.fit(log.filter(sf.col("item_id") != "item1"))
        # исходное количество пар - 3
        pred = model.predict_pairs(
            log.filter(sf.col("user_id") == "user1").select(
                "user_id", "item_id"
            )
        )
        # для холодного объекта не возвращаем ничего
        assert pred.count() == 2
        assert pred.select("user_id").distinct().collect()[0][0] == "user1"
        # предсказываем для всех теплых объектов
        assert list(
            pred.select("item_id")
            .distinct()
            .toPandas()
            .sort_values("item_id")["item_id"]
        ) == ["item2", "item3"]
    except:  # noqa
        pytest.fail()
