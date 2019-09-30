from typing import Set, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf


def get_distinct_values_in_column(df: DataFrame, column: str) -> Set[Any]:
    """
    возвращает уникальные значения в колонке spark-датафрейма в виде python set

    :param df: spark-датафрейм
    :param column: имя колонки
    :return: уникальные значения в колонке
    """
    return set([row[column]
                for row in (df
                            .select(column)
                            .distinct()
                            .collect())
                ])


def get_top_k_rows(df: DataFrame, k: int, sort_column: str):
    """

    :param sort_column:
    :param df:
    :param k:
    :return:
    """
    window = (Window
              .orderBy(df[sort_column].desc()))

    return (df
            .withColumn('rank',
                        sf.row_number().over(window))
            .filter(sf.col('rank') <= k)
            .drop('rank'))
