# pylint: disable-all
import pandas as pd

from replay.utils import fallback, convert2spark


def test_fallback():
    base = pd.DataFrame({"user_id": [1], "item_id": [1], "relevance": [1]})
    extra = pd.DataFrame(
        {"user_id": [1, 1, 2], "item_id": [1, 2, 1], "relevance": [1, 2, 1]}
    )
    base = convert2spark(base)
    extra = convert2spark(extra)
    res = fallback(base, extra, 2).toPandas()
    assert len(res) == 3
    assert res.user_id.nunique() == 2
    a = res.loc[
        (res["user_id"] == 1) & (res["item_id"] == 1), "relevance"
    ].iloc[0]
    b = res.loc[
        (res["user_id"] == 1) & (res["item_id"] == 2), "relevance"
    ].iloc[0]
    assert a > b
