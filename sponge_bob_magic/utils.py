"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Any, Optional, Set

import numpy as np
from pyspark.ml.linalg import DenseVector, Vector, Vectors, VectorUDT
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.sql.functions import udf
from scipy.sparse import csr_matrix


def get_distinct_values_in_column(
    dataframe: DataFrame, column: str
) -> Set[Any]:
    """
    Возвращает уникальные значения в колонке спарк-датафрейма в виде set.

    :param dataframe: spark-датафрейм
    :param column: имя колонки
    :return: уникальные значения в колонке
    """
    return {
        row[column] for row in (dataframe.select(column).distinct().collect())
    }


def func_get(vector: np.ndarray, i: int) -> float:
    """
    вспомогательная функция для создания Spark UDF для получения элемента
    массива по индексу

    :param vector: массив (vector в типах Scala или numpy array в PySpark)
    :param i: индекс, по которому нужно извлечь значение из массива
    :returns: значение ячейки массива (вещественное число)
    """
    return float(vector[i])


def get_top_k_recs(recs: DataFrame, k: int) -> DataFrame:
    """
    Выбирает из рекомендаций топ-k штук на основе `relevance`.

    :param recs: рекомендации, спарк-датафрейм с колонками
        `[user_id, item_id, relevance]`
    :param k: число рекомендаций для каждого юзера
    :return: топ-k рекомендации, спарк-датафрейм с колонками
        `[user_id, item_id, relevance]`
    """
    window = Window.partitionBy(recs["user_id"]).orderBy(
        recs["relevance"].desc()
    )
    return (
        recs.withColumn("rank", sf.row_number().over(window))
        .filter(sf.col("rank") <= k)
        .drop("rank")
    )


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
    >>> input_data.dtypes
    [('one', 'vector'), ('two', 'vector')]
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
    >>> input_data.dtypes
    [('one', 'vector'), ('two', 'vector')]
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
    return ", ".join(
        [
            f"total lines: {cnt}",
            f"total users: {user_cnt}",
            f"total items: {item_cnt}",
        ]
    )


def to_csr(
    log: DataFrame,
    user_count: Optional[int] = None,
    item_count: Optional[int] = None,
) -> csr_matrix:
    """
    Конвертирует лог в csr матрицу item-user.

    >>> import pandas as pd
    >>> from sponge_bob_magic.converter import convert
    >>> data_frame = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 2], "relevance": [1, 2]})
    >>> data_frame = convert(data_frame)
    >>> m = to_csr(data_frame).T
    >>> m.toarray()
    array([[1, 0],
           [0, 0],
           [0, 2]], dtype=int64)

    :param log: spark DataFrame с колонками ``user_id``, ``item_id`` и ``relevance``
    :param user_count: количество строк в результирующей матрице (если пусто, то вычисляется по логу)
    :param item_count: количество столбцов в результирующей матрице (если пусто, то вычисляется по логу)
    """
    pandas_df = log.select("user_idx", "item_idx", "relevance").toPandas()
    row_count = int(
        user_count if user_count is not None else pandas_df.user_idx.max() + 1
    )
    col_count = int(
        item_count if item_count is not None else pandas_df.item_idx.max() + 1
    )
    return csr_matrix(
        (pandas_df.relevance, (pandas_df.user_idx, pandas_df.item_idx)),
        shape=(row_count, col_count),
    )
