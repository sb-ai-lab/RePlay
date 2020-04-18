"""
Содержит функции, позволяющие отобрать данные по некоторому критерию.
"""
from typing import Union
import pandas as pd
import pyspark.sql as sp

from sponge_bob_magic.converter import TypeManager

AnyDataFrame = Union[sp.DataFrame, pd.DataFrame]


def min_entries(df: AnyDataFrame, n: int):
    """
    Удаляет из датафрейма записи всех пользователей,
    имеющих менее ``n`` оценок.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2]})
    >>> min_entries(df, 2)
       user_id
    0        1
    1        1
    """
    tm = TypeManager()
    df = tm.fit_convert(df)

    vc = df.groupBy("user_id").count()
    remaining_users = vc.filter(vc["count"] > n)[["user_id"]]
    df = df.join(remaining_users, on="user_id", how="inner")

    return tm.inverse(df)
