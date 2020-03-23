"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
from typing import Any, Iterable, List, Set, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as sf


def flat_list(list_object: Iterable):
    """
    Генератор.
    Из неоднородного листа с вложенными листами делает однородный лист.
    Например, [1, [2], [3, 4], 5] -> [1, 2, 3, 4, 5].

    :param list_object: лист
    :return: преобразованный лист
    """
    for item in list_object:
        if (
                isinstance(item, collections.abc.Iterable) and
                not isinstance(item, (str, bytes))
        ):
            yield from flat_list(item)
        else:
            yield item


def get_distinct_values_in_column(
        dataframe: DataFrame,
        column: str
) -> Set[Any]:
    """
    Возвращает уникальные значения в колонке спарк-датафрейма в виде set.

    :param dataframe: spark-датафрейм
    :param column: имя колонки
    :return: уникальные значения в колонке
    """
    return {
        row[column]
        for row in (dataframe
                    .select(column)
                    .distinct()
                    .collect())
    }


def get_top_k_rows(
        dataframe: DataFrame, k: int, sort_column: str
) -> DataFrame:
    """
    Выделяет топ-k строк в датафрейме на основе заданной колонки.

    :param sort_column: название колонки, по которой необходимы выделить топ
    :param dataframe: спарк-датафрейм
    :param k: сколько топовых строк необходимо выделить
    :return: спарк-датафрейм такого же вида, но размера `k`
    """
    window = (Window
              .orderBy(dataframe[sort_column].desc()))
    return (dataframe
            .withColumn("rank",
                        sf.row_number().over(window))
            .filter(sf.col("rank") <= k)
            .drop("rank"))


def func_get(vector: np.ndarray, i: int) -> float:
    """
    вспомогательная функция для создания Spark UDF для получения элемента
    массива по индексу

    :param vector: массив (vector в типах Scala или numpy array в PySpark)
    :param i: индекс, по которому нужно извлечь значение из массива
    :returns: значение ячейки массива (вещественное число)
    """
    return float(vector[i])


def get_feature_cols(
        user_features: DataFrame,
        item_features: DataFrame
) -> Tuple[List[str], List[str]]:
    """
    извлечь список свойств пользователей и объектов

    :param user_features: свойства пользователей в стандартном формате
    :param item_features: свойства объектов в стандартном формате
    :return: пара списков
    (имена колонок свойств пользователей, то же для фичей)
    """
    user_feature_cols = list(
        set(user_features.columns) - {"user_id", "timestamp"}
    )
    item_feature_cols = list(
        set(item_features.columns) - {"item_id", "timestamp"}
    )
    return user_feature_cols, item_feature_cols


def get_top_k_recs(recs: DataFrame, k: int) -> DataFrame:
    """
    Выбирает из рекомендаций топ-k штук на основе `relevance`.

    :param recs: рекомендации, спарк-датафрейм с колонками
        `[user_id, item_id, relevance]`
    :param k: число рекомендаций для каждого юзера
    :return: топ-k рекомендации, спарк-датафрейм с колонками
        `[user_id, item_id, relevance]`
    """
    window = (Window
              .partitionBy(recs["user_id"])
              .orderBy(recs["relevance"].desc()))
    return (recs
            .withColumn("rank",
                        sf.row_number().over(window))
            .filter(sf.col("rank") <= k)
            .drop("rank"))
