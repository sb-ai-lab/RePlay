"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Any, Set, Optional

from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql import functions as sf


def get_distinct_values_in_column(
        dataframe: DataFrame,
        column: str
) -> Set[Any]:
    """
    Возвращает уникальные значения в колонке спарк-датафрейма в виде set.

    :param dataframe: спарк-датафрейм
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
    Выделяет топ-k строк в датафрейме на оснвое заданной колонки.

    :param sort_column: название колонки, по которой необходимы выделить топ
    :param dataframe: спарк-датафрейм
    :param k: сколько топовых строк необходимо выделить
    :return: спарк-датафрейм такого же вида, но размера `k`
    """
    window = (Window
              .orderBy(dataframe[sort_column].desc()))

    return (dataframe
            .withColumn('rank',
                        sf.row_number().over(window))
            .filter(sf.col('rank') <= k)
            .drop('rank'))


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
