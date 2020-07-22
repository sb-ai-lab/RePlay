"""
Содержит функции, позволяющие отобрать данные по некоторому критерию.
"""
from replay.constants import AnyDataFrame
from replay.converter import convert


def min_entries(data_frame: AnyDataFrame, num_entries: int) -> AnyDataFrame:
    """
    Удаляет из датафрейма записи всех пользователей,
    имеющих менее ``num_entries`` оценок.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1, 1, 2]})
    >>> min_entries(data_frame, 2)
       user_id
    0        1
    1        1
    """
    type_in = type(data_frame)
    data_frame = convert(data_frame)
    entries_by_user = data_frame.groupBy("user_id").count()  # type: ignore
    remaining_users = entries_by_user.filter(
        entries_by_user["count"] >= num_entries
    )[["user_id"]]
    data_frame = data_frame.join(
        remaining_users, on="user_id", how="inner"
    )  # type: ignore
    return convert(data_frame, to_type=type_in)


def min_rating(
    data_frame: AnyDataFrame, value: float, column="relevance"
) -> AnyDataFrame:
    """
    Удаляет из датафрейма записи с оценкой меньше ``value`` в колонке ``column``.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"relevance": [1, 5, 3, 4]})
    >>> min_rating(data_frame, 3.5)
       relevance
    0          5
    1          4
    """
    type_in = type(data_frame)
    data_frame = convert(data_frame)
    data_frame = data_frame.filter(data_frame[column] > value)  # type: ignore
    return convert(data_frame, to_type=type_in)
