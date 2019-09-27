from typing import Set, Any

from pyspark.sql import DataFrame


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
