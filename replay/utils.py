"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Any, List, Optional, Set, Union

import numpy as np
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql.functions import col, element_at, row_number, udf
from pyspark.sql.types import DoubleType
from scipy.sparse import csr_matrix

from replay.constants import NumType


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
        recs.withColumn("rank", row_number().over(window))
        .filter(col("rank") <= k)
        .drop("rank")
    )


@udf(returnType=DoubleType())  # type: ignore
def vector_dot(one: DenseVector, two: DenseVector) -> float:
    """
    вычисляется скалярное произведение двух колонок-векторов

    >>> from replay.session_handler import State
    >>> from pyspark.ml.linalg import Vectors
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
    StructType(List(StructField(dot,DoubleType,true)))
    >>> output_data.show()
    +----+
    | dot|
    +----+
    |11.0|
    +----+
    <BLANKLINE>

    :param one: правый множитель-вектор
    :param two: левый множитель-вектор
    :returns: вектор с одним значением --- скалярным произведением
    """
    return float(one.dot(two))


@udf(returnType=VectorUDT())  # type: ignore
def vector_mult(
    one: Union[DenseVector, NumType], two: DenseVector
) -> DenseVector:
    """
    вычисляется покоординатное произведение двух колонок-векторов

    >>> from replay.session_handler import State
    >>> from pyspark.ml.linalg import Vectors
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

    >>> from replay.session_handler import State
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
    Конвертирует лог в csr матрицу user-item.

    >>> import pandas as pd
    >>> from replay.converter import convert
    >>> data_frame = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 2], "relevance": [1, 2]})
    >>> data_frame = convert(data_frame)
    >>> m = to_csr(data_frame)
    >>> m.toarray()
    array([[1, 0, 0],
           [0, 0, 2]])

    :param log: spark DataFrame с колонками ``user_id``, ``item_id`` и ``relevance``
    :param user_count: количество строк в результирующей матрице (если пусто, то вычисляется по логу)
    :param item_count: количество столбцов в результирующей матрице (если пусто, то вычисляется по логу)
    """
    pandas_df = log.select("user_idx", "item_idx", "relevance").toPandas()
    row_count = int(
        user_count
        if user_count is not None
        else pandas_df["user_idx"].max() + 1
    )
    col_count = int(
        item_count
        if item_count is not None
        else pandas_df["item_idx"].max() + 1
    )
    return csr_matrix(
        (
            pandas_df["relevance"],
            (pandas_df["user_idx"], pandas_df["item_idx"]),
        ),
        shape=(row_count, col_count),
    )


def horizontal_explode(
    data_frame: DataFrame,
    column_to_explode: str,
    prefix: str,
    other_columns: List[Column],
) -> DataFrame:
    """
    аналог функции ``explode``, только одну колонку с массивом значений разбивает на несколько.
    В каждой строке разбиваемой колонки должно быть одинаковое количество значений

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([(5, [1.0, 2.0]), (6, [3.0, 4.0])])
    ...     .toDF("id_col", "array_col")
    ... )
    >>> input_data.show()
    +------+----------+
    |id_col| array_col|
    +------+----------+
    |     5|[1.0, 2.0]|
    |     6|[3.0, 4.0]|
    +------+----------+
    <BLANKLINE>
    >>> horizontal_explode(input_data, "array_col", "element", [col("id_col")]).show()
    +------+---------+---------+
    |id_col|element_0|element_1|
    +------+---------+---------+
    |     5|      1.0|      2.0|
    |     6|      3.0|      4.0|
    +------+---------+---------+
    <BLANKLINE>

    :param data_frame: spark DataFrame, в котором нужно разбить колонку
    :param column_to_explode: колонка типа ``array``, которую нужно разбить
    :param prefix: на выходе будут колонки с именами ``prefix_0, prefix_1`` и т. д.
    :param other_columns: список колонок, которые нужно сохранить в выходном DataFrame помимо разбиваемой
    :returns: DataFrame с колонками, порождёнными элементами ``column_to_explode``
    """
    num_columns = len(data_frame.select(column_to_explode).head()[0])
    return data_frame.select(
        *other_columns,
        *[
            element_at(column_to_explode, i + 1).alias(f"{prefix}_{i}")
            for i in range(num_columns)
        ],
    )
