"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
from typing import Any, Iterable, List, Set, Tuple

import numpy as np
from pyspark.ml.linalg import DenseVector, Vector, Vectors, VectorUDT
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.sql.functions import udf


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


@udf(returnType=VectorUDT())
def to_vector(array: List[float]) -> DenseVector:
    """
    преобразует список вещественных значений в плотный вектор в формате Spark

    >>> from sponge_bob_magic.session_handler import State
    >>> spark = State().session
    >>> input_data = spark.createDataFrame([([1.0, 2.0, 3.0],)]).toDF("array")
    >>> input_data.schema
    StructType(List(StructField(array,ArrayType(DoubleType,true),true)))
    >>> input_data.show()
    +---------------+
    |          array|
    +---------------+
    |[1.0, 2.0, 3.0]|
    +---------------+
    <BLANKLINE>
    >>> output_data = input_data.select(to_vector("array").alias("vector"))
    >>> output_data.schema
    StructType(List(StructField(vector,VectorUDT,true)))
    >>> output_data.show()
    +-------------+
    |       vector|
    +-------------+
    |[1.0,2.0,3.0]|
    +-------------+
    <BLANKLINE>

    :param array: список вещественных чисел
    :returns: плотный вектор из пакета ``pyspark.ml.linalg``
    """
    return Vectors.dense(array)


@udf(returnType=VectorUDT())
def vector_dot(one: Vector, two: Vector) -> DenseVector:
    """
    вычисляется скалярное произведение двух колонок-векторов

    >>> from sponge_bob_magic.session_handler import State
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([(Vectors.dense([1.0, 2.0]), Vectors.dense([3.0, 4.0]))])
    ...     .toDF("one", "two")
    ... )
    >>> input_data.schema
    StructType(List(StructField(one,VectorUDT,true),StructField(two,VectorUDT,true)))
    >>> input_data.show()
    +---------+---------+
    |      one|      two|
    +---------+---------+
    |[1.0,2.0]|[3.0,4.0]|
    +---------+---------+
    <BLANKLINE>
    >>> output_data = input_data.select(vector_dot("one", "two").alias("dot"))
    >>> output_data.schema
    StructType(List(StructField(dot,VectorUDT,true)))
    >>> output_data.show()
    +------+
    |   dot|
    +------+
    |[11.0]|
    +------+
    <BLANKLINE>

    :param one: правый множитель-вектор
    :param two: левый множитель-вектор
    :returns: вектор с одним значением --- скалярным произведением
    """
    return Vectors.dense([one.dot(two)])


@udf(returnType=VectorUDT())
def vector_mult(one: Vector, two: Vector) -> DenseVector:
    """
    вычисляется покоординатное произведение двух колонок-векторов

    >>> from sponge_bob_magic.session_handler import State
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([(Vectors.dense([1.0, 2.0]), Vectors.dense([3.0, 4.0]))])
    ...     .toDF("one", "two")
    ... )
    >>> input_data.schema
    StructType(List(StructField(one,VectorUDT,true),StructField(two,VectorUDT,true)))
    >>> input_data.show()
    +---------+---------+
    |      one|      two|
    +---------+---------+
    |[1.0,2.0]|[3.0,4.0]|
    +---------+---------+
    <BLANKLINE>
    >>> output_data = input_data.select(vector_mult("one", "two").alias("mult"))
    >>> output_data.schema
    StructType(List(StructField(mult,VectorUDT,true)))
    >>> output_data.show()
    +---------+
    |     mult|
    +---------+
    |[3.0,8.0]|
    +---------+
    <BLANKLINE>

    :param one: правый множитель-вектор
    :param two: левый множитель-вектор
    :returns: вектор с результатом покоординатного умножения
    """
    return one * two


def get_log_info(log: DataFrame) -> str:
    """
    простейшая статистика по логу предпочтений пользователей

    >>> from sponge_bob_magic.session_handler import State
    >>> spark = State().session
    >>> log = spark.createDataFrame([(1, 2), (3, 4), (5, 2)]).toDF("user_id", "item_id")
    >>> log.show()
    +-------+-------+
    |user_id|item_id|
    +-------+-------+
    |      1|      2|
    |      3|      4|
    |      5|      2|
    +-------+-------+
    <BLANKLINE>
    >>> get_log_info(log)
    'total lines: 3, total users: 3, total items: 2'

    :param log: таблица с колонками ``user_id`` и ``item_id``
    :returns: строку со статистикой
    """
    cnt = log.count()
    user_cnt = log.select("user_id").distinct().count()
    item_cnt = log.select("item_id").distinct().count()
    return ", ".join([
        f"total lines: {cnt}",
        f"total users: {user_cnt}",
        f"total items: {item_cnt}"
    ])
