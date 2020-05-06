"""
Содержит функции, позволяющие отобрать данные по некоторому критерию.
"""
from typing import Union
import pandas as pd
import pyspark.sql as sp

from sponge_bob_magic.converter import get_type, convert

AnyDataFrame = Union[sp.DataFrame, pd.DataFrame]


def min_entries(df: AnyDataFrame, n: int) -> AnyDataFrame:
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
    type_in = get_type(df)
    df = convert(df)

    vc = df.groupBy("user_id").count()
    remaining_users = vc.filter(vc["count"] >= n)[["user_id"]]
    df = df.join(remaining_users, on="user_id", how="inner")

    return convert(df, to=type_in)


def min_rating(
    df: AnyDataFrame, value: float, column="relevance"
) -> AnyDataFrame:
    """
    Удаляет из датафрейма записи с оценкой меньше ``value`` в колонке ``column``.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"relevance": [1, 5, 3, 4]})
    >>> min_rating(df, 3.5)
       relevance
    0          5
    1          4
    """
    type_in = get_type(df)
    df = convert(df)

    df = df.filter(df[column] > value)

    return convert(df, to=type_in)
