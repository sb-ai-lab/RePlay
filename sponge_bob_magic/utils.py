"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
from typing import Any, Set

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf


def get_distinct_values_in_column(
        dataframe: DataFrame,
        column: str
) -> Set[Any]:
    """
    возвращает уникальные значения в колонке spark-датафрейма в виде python set

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

    :param sort_column:
    :param dataframe:
    :param k:
    :return:
    """
    window = (Window
              .orderBy(dataframe[sort_column].desc()))

    return (dataframe
            .withColumn('rank',
                        sf.row_number().over(window))
            .filter(sf.col('rank') <= k)
            .drop('rank'))
