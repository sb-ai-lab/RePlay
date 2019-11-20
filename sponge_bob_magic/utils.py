"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Any, List, Optional, Set, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as sf
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


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


def get_top_k_rows(dataframe: DataFrame, k: int, sort_column: str):
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


def write_read_dataframe(spark: SparkSession,
                         df: DataFrame,
                         path: Optional[str]):
    """
    Записывает спарк-датафрейм на диск и считывает его обратно и возвращает.
    Если путь равен None, то возвращается спарк-датафрейм, поданный на вход.

    :param spark: инициализированная спарк-сессия
    :param df: спарк-датафрейм
    :param path: путь, по которому происходит записаь датафрейма
    :return: оригинальный датафрейм; если `path` не пустой,
        то lineage датафрейма обнуляется
    """
    if path is not None:
        df.write.parquet(path)
        df = spark.read.parquet(path)
    return df


def func_get(vector: np.ndarray, i: int) -> float:
    """
    вспомогательная функция для создания Spark UDF для получения элемента
    массива по индексу

    :param vector: массив (vector в типах Scala или numpy array в PySpark)
    :param i: индекс, по которому нужно извлечь значение из массива
    :returns: значение ячейки массива (вещественное число)
    """
    return float(vector[i])


udf_get = udf(func_get, DoubleType())


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
