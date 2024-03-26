import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from replay.models import ItemKNN
from replay.scenarios import Fallback
from replay.utils.spark_utils import convert2spark, fallback
from tests.utils import create_dataset, sparkDataFrameEqual


@pytest.mark.spark
def test_fallback():
    base = pd.DataFrame({"user_idx": [1], "item_idx": [1], "relevance": [1]})
    extra = pd.DataFrame({"user_idx": [1, 1, 2], "item_idx": [1, 2, 1], "relevance": [1, 2, 1]})
    base = convert2spark(base)
    extra = convert2spark(extra)
    res = fallback(base, extra, 2).toPandas()
    assert len(res) == 3
    assert res.user_idx.nunique() == 2
    a = res.loc[(res["user_idx"] == 1) & (res["item_idx"] == 1), "relevance"].iloc[0]
    b = res.loc[(res["user_idx"] == 1) & (res["item_idx"] == 2), "relevance"].iloc[0]
    assert a > b
    bypass_res = fallback(base, None, 2)
    sparkDataFrameEqual(bypass_res, base)


@pytest.mark.spark
def test_class(log, log2):
    model = Fallback(ItemKNN(), threshold=3)
    assert model._init_args == {"threshold": 3}
    s = str(model)
    assert s == "Fallback_ItemKNN_PopRec"
    dataset = create_dataset(log)
    dataset2 = create_dataset(log2)
    model.fit(dataset2)
    p1, p2 = model.optimize(dataset, dataset2, k=1, budget=1)
    assert p2 is None
    assert isinstance(p1, dict)
    model.predict(dataset2, k=1)

    model = Fallback(ItemKNN(), ItemKNN(), threshold=3)
    p1, p2 = model.optimize(dataset, dataset2, k=1, budget=1)
    assert isinstance(p1, dict)
    assert isinstance(p2, dict)
