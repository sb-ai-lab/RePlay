import collections
import logging
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing import Iterable

import numpy as np
import pandas as pd
import pyspark.sql.types as st

from numpy.random import default_rng
from pyarrow import fs
from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
from pyspark.sql import SparkSession, Column, DataFrame, Window, functions as sf
from scipy.sparse import csr_matrix

from replay.constants import AnyDataFrame, NumType, REC_SCHEMA
from replay.session_handler import State
from pyspark.sql.column import _to_java_column, _to_seq

# pylint: disable=invalid-name

logger = logging.getLogger("replay")


def convert2spark(data_frame: Optional[AnyDataFrame]) -> Optional[DataFrame]:
    """
    Converts Pandas DataFrame to Spark DataFrame

    :param data_frame: pandas DataFrame
    :return: converted data
    """
    if data_frame is None:
        return None
    if isinstance(data_frame, DataFrame):
        return data_frame
    spark = State().session
    return spark.createDataFrame(data_frame)  # type: ignore


def get_distinct_values_in_column(
    dataframe: DataFrame, column: str
) -> Set[Any]:
    """
    Get unique values from a column as a set.

    :param dataframe: spark DataFrame
    :param column: column name
    :return: set of unique values
    """
    return {
        row[column] for row in (dataframe.select(column).distinct().collect())
    }


def func_get(vector: np.ndarray, i: int) -> float:
    """
    helper function for Spark UDF to get element by index

    :param vector: Scala vector or numpy array
    :param i: index in a vector
    :returns: element value
    """
    return float(vector[i])


def get_top_k(
    dataframe: DataFrame,
    partition_by_col: Column,
    order_by_col: List[Column],
    k: int,
) -> DataFrame:
    """
    Return top ``k`` rows for each entity in ``partition_by_col`` ordered by
    ``order_by_col``.

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> log = spark.createDataFrame([(1, 2, 1.), (1, 3, 1.), (1, 4, 0.5), (2, 1, 1.)]).toDF("user_id", "item_id", "relevance")
    >>> log.show()
    +-------+-------+---------+
    |user_id|item_id|relevance|
    +-------+-------+---------+
    |      1|      2|      1.0|
    |      1|      3|      1.0|
    |      1|      4|      0.5|
    |      2|      1|      1.0|
    +-------+-------+---------+
    <BLANKLINE>
    >>> get_top_k(dataframe=log,
    ...    partition_by_col=sf.col('user_id'),
    ...    order_by_col=[sf.col('relevance').desc(), sf.col('item_id').desc()],
    ...    k=1).orderBy('user_id').show()
    +-------+-------+---------+
    |user_id|item_id|relevance|
    +-------+-------+---------+
    |      1|      3|      1.0|
    |      2|      1|      1.0|
    +-------+-------+---------+
    <BLANKLINE>

    :param dataframe: spark dataframe to filter
    :param partition_by_col: spark column to partition by
    :param order_by_col: list of spark columns to orted by
    :param k: number of first rows for each entity in ``partition_by_col`` to return
    :return: filtered spark dataframe
    """
    return (
        dataframe.withColumn(
            "temp_rank",
            sf.row_number().over(
                Window.partitionBy(partition_by_col).orderBy(*order_by_col)
            ),
        )
        .filter(sf.col("temp_rank") <= k)
        .drop("temp_rank")
    )


def get_top_k_recs(recs: DataFrame, k: int, id_type: str = "idx") -> DataFrame:
    """
    Get top k recommendations by `relevance`.

    :param recs: recommendations DataFrame
        `[user_id, item_id, relevance]`
    :param k: length of a recommendation list
    :param id_type: id or idx
    :return: top k recommendations `[user_id, item_id, relevance]`
    """
    return get_top_k(
        dataframe=recs,
        partition_by_col=sf.col(f"user_{id_type}"),
        order_by_col=[sf.col("relevance").desc()],
        k=k,
    )


def delete_folder(path: str):
    file_info = get_filesystem(path)

    if file_info.filesystem == FileSystem.HDFS:
        fs.HadoopFileSystem.from_uri(file_info.hdfs_uri).delete_dir(path)
    else:
        fs.LocalFileSystem().delete_dir(file_info.path)


def create_folder(path: str, delete_if_exists: bool = False, exists_ok: bool = False):
    file_info = get_filesystem(path)

    is_exists = do_path_exists(path)
    if is_exists and delete_if_exists:
        delete_folder(path)
    elif is_exists and not exists_ok:
        raise FileExistsError(f"The path already exists: {path}")

    if file_info.filesystem == FileSystem.HDFS:
        fs.HadoopFileSystem.from_uri(file_info.hdfs_uri).create_dir(file_info.path)
    else:
        fs.LocalFileSystem().create_dir(file_info.path)


@sf.udf(returnType=st.DoubleType())
def vector_dot(one: DenseVector, two: DenseVector) -> float:
    """
    dot product of two column vectors

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

    :param one: vector one
    :param two: vector two
    :returns: dot product
    """
    return float(one.dot(two))


@sf.udf(returnType=VectorUDT())  # type: ignore
def vector_mult(
    one: Union[DenseVector, NumType], two: DenseVector
) -> DenseVector:
    """
    elementwise vector multiplication

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

    :param one: vector one
    :param two: vector two
    :returns: result
    """
    return one * two


def multiply_scala_udf(scalar, vector):
    """Multiplies a scalar by a vector

    Args:
        scalar: column with scalars
        vector: column with vectors

    Returns: column expression
    """
    sc = SparkSession.getActiveSession().sparkContext
    _f = sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.multiplyUDF()
    return Column(_f.apply(_to_seq(sc, [scalar, vector], _to_java_column)))


@sf.udf(returnType=st.ArrayType(st.DoubleType()))
def array_mult(first: st.ArrayType, second: st.ArrayType):
    """
    elementwise array multiplication

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([([1.0, 2.0], [3.0, 4.0])])
    ...     .toDF("one", "two")
    ... )
    >>> input_data.dtypes
    [('one', 'array<double>'), ('two', 'array<double>')]
    >>> input_data.show()
    +----------+----------+
    |       one|       two|
    +----------+----------+
    |[1.0, 2.0]|[3.0, 4.0]|
    +----------+----------+
    <BLANKLINE>
    >>> output_data = input_data.select(array_mult("one", "two").alias("mult"))
    >>> output_data.schema
    StructType(List(StructField(mult,ArrayType(DoubleType,true),true)))
    >>> output_data.show()
    +----------+
    |      mult|
    +----------+
    |[3.0, 8.0]|
    +----------+
    <BLANKLINE>

    :param first: first array
    :param second: second array
    :returns: result
    """

    return [first[i] * second[i] for i in range(len(first))]


def get_log_info(
    log: DataFrame, user_col="user_idx", item_col="item_idx"
) -> str:
    """
    Basic log statistics

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> log = spark.createDataFrame([(1, 2), (3, 4), (5, 2)]).toDF("user_idx", "item_idx")
    >>> log.show()
    +--------+--------+
    |user_idx|item_idx|
    +--------+--------+
    |       1|       2|
    |       3|       4|
    |       5|       2|
    +--------+--------+
    <BLANKLINE>
    >>> get_log_info(log)
    'total lines: 3, total users: 3, total items: 2'

    :param log: interaction log containing ``user_idx`` and ``item_idx``
    :param user_col: name of a columns containing users' identificators
    :param item_col: name of a columns containing items' identificators

    :returns: statistics string
    """
    cnt = log.count()
    user_cnt = log.select(user_col).distinct().count()
    item_cnt = log.select(item_col).distinct().count()
    return ", ".join(
        [
            f"total lines: {cnt}",
            f"total users: {user_cnt}",
            f"total items: {item_cnt}",
        ]
    )


def get_stats(
    log: DataFrame, group_by: str = "user_id", target_column: str = "relevance"
) -> DataFrame:
    """
    Calculate log statistics: min, max, mean, median ratings, number of ratings.
    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> test_df = (spark.
    ...   createDataFrame([(1, 2, 1), (1, 3, 3), (1, 1, 2), (2, 3, 2)])
    ...   .toDF("user_id", "item_id", "rel")
    ...   )
    >>> get_stats(test_df, target_column='rel').show()
    +-------+--------+-------+-------+---------+----------+
    |user_id|mean_rel|max_rel|min_rel|count_rel|median_rel|
    +-------+--------+-------+-------+---------+----------+
    |      1|     2.0|      3|      1|        3|         2|
    |      2|     2.0|      2|      2|        1|         2|
    +-------+--------+-------+-------+---------+----------+
    <BLANKLINE>
    >>> get_stats(test_df, group_by='item_id', target_column='rel').show()
    +-------+--------+-------+-------+---------+----------+
    |item_id|mean_rel|max_rel|min_rel|count_rel|median_rel|
    +-------+--------+-------+-------+---------+----------+
    |      2|     1.0|      1|      1|        1|         1|
    |      3|     2.5|      3|      2|        2|         2|
    |      1|     2.0|      2|      2|        1|         2|
    +-------+--------+-------+-------+---------+----------+
    <BLANKLINE>

    :param log: spark DataFrame with ``user_id``, ``item_id`` and ``relevance`` columns
    :param group_by: column to group data by, ``user_id`` or ``item_id``
    :param target_column: column with interaction ratings
    :return: spark DataFrame with statistics
    """
    agg_functions = {
        "mean": sf.avg,
        "max": sf.max,
        "min": sf.min,
        "count": sf.count,
    }
    agg_functions_list = [
        func(target_column).alias(str(name + "_" + target_column))
        for name, func in agg_functions.items()
    ]
    agg_functions_list.append(
        sf.expr(f"percentile_approx({target_column}, 0.5)").alias(
            "median_" + target_column
        )
    )

    return log.groupBy(group_by).agg(*agg_functions_list)


def check_numeric(feature_table: DataFrame) -> None:
    """
    Check if spark DataFrame columns are of NumericType
    :param feature_table: spark DataFrame
    """
    for column in feature_table.columns:
        if not isinstance(
            feature_table.schema[column].dataType, st.NumericType
        ):
            raise ValueError(
                f"""Column {column} has type {feature_table.schema[
            column].dataType}, that is not numeric."""
            )


def to_csr(
    log: DataFrame,
    user_count: Optional[int] = None,
    item_count: Optional[int] = None,
) -> csr_matrix:
    """
    Convert DataFrame to csr matrix

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 2], "relevance": [1, 2]})
    >>> data_frame = convert2spark(data_frame)
    >>> m = to_csr(data_frame)
    >>> m.toarray()
    array([[1, 0, 0],
           [0, 0, 2]])

    :param log: interaction log with ``user_idx``, ``item_idx`` and
    ``relevance`` columns
    :param user_count: number of rows in resulting matrix
    :param item_count: number of columns in resulting matrix
    """
    pandas_df = log.select("user_idx", "item_idx", "relevance").toPandas()
    if pandas_df.empty:
        return csr_matrix(
            (
                [],
                ([],[]),
            ),
            shape=(0, 0),
        )

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
    Transform a column with an array of values into separate columns.
    Each array must contain the same amount of values.

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
    >>> horizontal_explode(input_data, "array_col", "element", [sf.col("id_col")]).show()
    +------+---------+---------+
    |id_col|element_0|element_1|
    +------+---------+---------+
    |     5|      1.0|      2.0|
    |     6|      3.0|      4.0|
    +------+---------+---------+
    <BLANKLINE>

    :param data_frame: input DataFrame
    :param column_to_explode: column with type ``array``
    :param prefix: prefix used for new columns, suffix is an integer
    :param other_columns: columns to select beside newly created
    :returns: DataFrame with elements from ``column_to_explode``
    """
    num_columns = len(data_frame.select(column_to_explode).head()[0])
    return data_frame.select(
        *other_columns,
        *[
            sf.element_at(column_to_explode, i + 1).alias(f"{prefix}_{i}")
            for i in range(num_columns)
        ],
    )


def join_or_return(first, second, on, how):
    """
    Safe wrapper for join of two DataFrames if ``second`` parameter is None it returns ``first``.

    :param first: Spark DataFrame
    :param second: Spark DataFrame
    :param on: name of the join column
    :param how: type of join
    :return: Spark DataFrame
    """
    if second is None:
        return first
    return first.join(second, on=on, how=how)


def fallback(
    base: DataFrame, fill: DataFrame, k: int, id_type: str = "idx"
) -> DataFrame:
    """
    Fill missing recommendations for users that have less than ``k`` recommended items.
    Score values for the fallback model may be decreased to preserve sorting.

    :param base: base recommendations that need to be completed
    :param fill: extra recommendations
    :param k: desired recommendation list lengths for each user
    :param id_type: id or idx
    :return: augmented recommendations
    """
    if fill is None:
        return base
    if base.count() == 0:
        return get_top_k_recs(fill, k, id_type)
    margin = 0.1
    min_in_base = base.agg({"relevance": "min"}).collect()[0][0]
    max_in_fill = fill.agg({"relevance": "max"}).collect()[0][0]
    diff = max_in_fill - min_in_base
    fill = fill.withColumnRenamed("relevance", "relevance_fallback")
    if diff >= 0:
        fill = fill.withColumn(
            "relevance_fallback", sf.col("relevance_fallback") - diff - margin
        )
    recs = base.join(
        fill, on=["user_" + id_type, "item_" + id_type], how="full_outer"
    )
    recs = recs.withColumn(
        "relevance", sf.coalesce("relevance", "relevance_fallback")
    ).select("user_" + id_type, "item_" + id_type, "relevance")
    recs = get_top_k_recs(recs, k, id_type)
    return recs


def cache_if_exists(dataframe: Optional[DataFrame]) -> Optional[DataFrame]:
    """
    Cache a DataFrame
    :param dataframe: Spark DataFrame or None
    :return: DataFrame or None
    """
    if dataframe is not None:
        return dataframe.cache()
    return dataframe


def unpersist_if_exists(dataframe: Optional[DataFrame]) -> None:
    """
    :param dataframe: DataFrame or None
    """
    if dataframe is not None and dataframe.is_cached:
        dataframe.unpersist()


def join_with_col_renaming(
    left: DataFrame,
    right: DataFrame,
    on_col_name: Union[str, List],
    how: str = "inner",
    suffix="join",
) -> DataFrame:
    """
    There is a bug in some Spark versions (e.g. 3.0.2), which causes errors
    in joins of DataFrames derived form the same DataFrame on the columns with the same name:
    https://issues.apache.org/jira/browse/SPARK-14948
    https://issues.apache.org/jira/browse/SPARK-36815.

    The function renames columns stated in `on_col_name` in one dataframe,
    performs join and removes renamed columns.

    :param left: left-side dataframe
    :param right: right-side dataframe
    :param on_col_name: column names to join on
    :param how: join type
    :param suffix: suffix added to `on_col_name` values to name temporary column
    :return: join result
    """
    if isinstance(on_col_name, str):
        on_col_name = [on_col_name]

    on_condition = sf.lit(True)
    for name in on_col_name:
        if how == "right":
            left = left.withColumnRenamed(name, f"{name}_{suffix}")
        else:
            right = right.withColumnRenamed(name, f"{name}_{suffix}")
        on_condition &= sf.col(name) == sf.col(f"{name}_{suffix}")

    return (left.join(right, on=on_condition, how=how)).drop(
        *[f"{name}_{suffix}" for name in on_col_name]
    )


def add_to_date(
    dataframe: DataFrame,
    column_name: str,
    base_date: str,
    base_date_format: Optional[str] = None,
) -> DataFrame:
    """
    Get user or item features from replay model.
    If a model can return both user and item embeddings,
    elementwise multiplication can be performed too.
    If a model can't return embedding for specific user/item, zero vector is returned.
    Treats column ``column_name`` as a number of days after the ``base_date``.
    Converts ``column_name`` to TimestampType with
    ``base_date`` + values of the ``column_name``.

    >>> from replay.session_handler import State
    >>> from pyspark.sql.types import IntegerType
    >>> spark = State().session
    >>> input_data = (
    ...     spark.createDataFrame([5, 6], IntegerType())
    ...     .toDF("days")
    ... )
    >>> input_data.show()
    +----+
    |days|
    +----+
    |   5|
    |   6|
    +----+
    <BLANKLINE>
    >>> add_to_date(input_data, 'days', '2021/09/01', 'yyyy/MM/dd').show()
    +-------------------+
    |               days|
    +-------------------+
    |2021-09-06 00:00:00|
    |2021-09-07 00:00:00|
    +-------------------+
    <BLANKLINE>

    :param dataframe: spark dataframe
    :param column_name: name of a column with numbers
        to add to the ``base_date``
    :param base_date: str with the date to add to
    :param base_date_format: base date pattern to parse
    :return: dataframe with new ``column_name`` converted to TimestampType
    """
    dataframe = (
        dataframe.withColumn(
            "tmp", sf.to_timestamp(sf.lit(base_date), format=base_date_format)
        )
        .withColumn(
            column_name,
            sf.to_timestamp(sf.expr(f"date_add(tmp, {column_name})")),
        )
        .drop("tmp")
    )
    return dataframe


def process_timestamp_column(
    dataframe: DataFrame,
    column_name: str,
    date_format: Optional[str] = None,
) -> DataFrame:
    """
    Convert ``column_name`` column of numeric/string/timestamp type
    to TimestampType.
    Return original ``dataframe`` if the column has TimestampType.
    Treats numbers as unix timestamp, treats strings as
    a string representation of dates in ``date_format``.
    Date format is inferred by pyspark if not defined by ``date_format``.

    :param dataframe: spark dataframe
    :param column_name: name of ``dataframe`` column to convert
    :param date_format: datetime pattern passed to
        ``to_timestamp`` pyspark sql function
    :return: dataframe with updated column ``column_name``
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column {column_name} not found")

    # no conversion needed
    if isinstance(dataframe.schema[column_name].dataType, st.TimestampType):
        return dataframe

    # unix timestamp
    if isinstance(dataframe.schema[column_name].dataType, st.NumericType):
        return dataframe.withColumn(
            column_name, sf.to_timestamp(sf.from_unixtime(sf.col(column_name)))
        )

    # datetime in string format
    dataframe = dataframe.withColumn(
        column_name,
        sf.to_timestamp(sf.col(column_name), format=date_format),
    )
    return dataframe


@sf.udf(returnType=VectorUDT())
def list_to_vector_udf(array: st.ArrayType) -> DenseVector:
    """
    convert spark array to vector

    :param array: spark Array to convert
    :return:  spark DenseVector
    """
    return Vectors.dense(array)


@sf.udf(returnType=st.FloatType())
def vector_squared_distance(first: DenseVector, second: DenseVector) -> float:
    """
    :param first: first vector
    :param second: second vector
    :returns: squared distance value
    """
    return float(first.squared_distance(second))


@sf.udf(returnType=st.FloatType())
def vector_euclidean_distance_similarity(
    first: DenseVector, second: DenseVector
) -> float:
    """
    :param first: first vector
    :param second: second vector
    :returns: 1/(1 + euclidean distance value)
    """
    return 1 / (1 + float(first.squared_distance(second)) ** 0.5)


@sf.udf(returnType=st.FloatType())
def cosine_similarity(first: DenseVector, second: DenseVector) -> float:
    """
    :param first: first vector
    :param second: second vector
    :returns: cosine similarity value
    """
    num = first.dot(second)
    denom = first.dot(first) ** 0.5 * second.dot(second) ** 0.5
    return float(num / denom)


def cache_temp_view(df: DataFrame, name: str) -> None:
    """
    Create Spark SQL temporary view with `name` and cache it
    """
    spark = State().session
    df.createOrReplaceTempView(name)
    spark.sql(f"cache table {name}")


def drop_temp_view(temp_view_name: str) -> None:
    """
    Uncache and drop Spark SQL temporary view
    """
    spark = State().session
    spark.catalog.dropTempView(temp_view_name)


class log_exec_timer:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._start = None
        self._duration = None

    def __enter__(self):
        self._start = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self._duration = (datetime.now() - self._start).total_seconds()
        msg = (
            f"Exec time of {self.name}: {self._duration}"
            if self.name
            else f"Exec time: {self._duration}"
        )
        logger.info(msg)

    @property
    def duration(self):
        return self._duration


@contextmanager
def JobGroup(group_id: str, description: str):
    sc = SparkSession.getActiveSession().sparkContext
    sc.setJobGroup(group_id, description)
    yield f"{group_id} - {description}"
    sc._jsc.clearJobGroup()


def cache_and_materialize_if_in_debug(df: DataFrame, description: str = "no-desc"):
    if os.environ.get("REPLAY_DEBUG_MODE", None):
        with log_exec_timer(description):
            df = df.cache()
            df.write.mode('overwrite').format('noop').save()


@contextmanager
def JobGroupWithMetrics(group_id: str, description: str):
    metric_name = f"{group_id}__{description}"
    with JobGroup(group_id, description), log_exec_timer(metric_name) as timer:
        yield

    if os.environ.get("REPLAY_DEBUG_MODE", None):
        import mlflow
        mlflow.log_metric(timer.name, timer.duration)


def get_number_of_allocated_executors(spark: SparkSession):
    sc = spark._jsc.sc()
    return (
        len(
            [
                executor.host()
                for executor in sc.statusTracker().getExecutorInfos()
            ]
        )
        - 1
    )


def get_full_class_name(instance) -> str:
    return ".".join([type(instance).__module__, type(instance).__name__])


def get_class_by_class_name(clazz: str) -> Any:
    """
        Loads Python class from its name.
    """
    parts = clazz.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


class FileSystem(Enum):
    HDFS = 1
    LOCAL = 2


def get_default_fs() -> str:
    spark = SparkSession.getActiveSession()
    hadoop_conf = spark._jsc.hadoopConfiguration()
    default_fs = hadoop_conf.get("fs.defaultFS")
    logger.debug(f"hadoop_conf.get('fs.defaultFS'): {default_fs}")
    return default_fs


@dataclass(frozen=True)
class FileInfo:
    path: str
    filesystem: FileSystem
    hdfs_uri: str = None


def get_filesystem(path: str) -> FileInfo:
    """Analyzes path and hadoop config and return tuple of `filesystem`,
    `hdfs uri` (if filesystem is hdfs) and `cleaned path` (without prefix).

    For example:

    >>> path = 'hdfs://node21.bdcl:9000/tmp/file'
    >>> get_filesystem(path)
    FileInfo(path='/tmp/file', filesystem=<FileSystem.HDFS: 1>, hdfs_uri='hdfs://node21.bdcl:9000')
    or
    >>> path = 'file:///tmp/file'
    >>> get_filesystem(path)
    FileInfo(path='/tmp/file', filesystem=<FileSystem.LOCAL: 2>, hdfs_uri=None)

    Args:
        path (str): path to file on hdfs or local disk

    Returns:
        Tuple[int, Optional[str], str]: `filesystem id`,
    `hdfs uri` (if filesystem is hdfs) and `cleaned path` (without prefix)
    """
    prefix_len = 7  # 'hdfs://' and 'file://' length
    if path.startswith("hdfs://"):
        if path.startswith("hdfs:///"):
            default_fs = get_default_fs()
            if default_fs.startswith("hdfs://"):
                return FileInfo(path[prefix_len:], FileSystem.HDFS, default_fs)
            else:
                raise Exception(
                    f"Can't get default hdfs uri for path = '{path}'. "
                    "Specify an explicit path, such as 'hdfs://host:port/dir/file', "
                    "or set 'fs.defaultFS' in hadoop configuration."
                )
        else:
            hostname = path[prefix_len:].split("/", 1)[0]
            hdfs_uri = "hdfs://" + hostname
            return FileInfo(path[len(hdfs_uri):], FileSystem.HDFS, hdfs_uri)
    elif path.startswith("file://"):
        return FileInfo(path[prefix_len:], FileSystem.LOCAL)
    else:
        default_fs = get_default_fs()
        if default_fs.startswith("hdfs://"):
            return FileInfo(path, FileSystem.HDFS, default_fs)
        else:
            return FileInfo(path, FileSystem.LOCAL)


def sample_top_k_recs(pairs: DataFrame, k: int, seed: int = None):
    """
    Sample k items for each user with probability proportional to the relevance score.

    Motivation: sometimes we have a pre-defined list of items for each user
    and could use `predict_pairs` method of RePlay models to score them.
    After that we could select top K most relevant items for each user
    with `replay.utils.get_top_k_recs` or sample them with
    probabilities proportional to their relevance score
    with `replay.utils.sample_top_k_recs` to get more diverse recommendations.

    :param pairs: spark dataframe with columns ``[user_idx, item_idx, relevance]``
    :param k: number of items for each user to return
    :param seed: random seed
    :return:  spark dataframe with columns ``[user_idx, item_idx, relevance]``
    """
    pairs = pairs.withColumn(
        "probability",
        sf.col("relevance")
        / sf.sum("relevance").over(Window.partitionBy("user_idx")),
    )

    def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
        user_idx = pandas_df["user_idx"][0]

        if seed is not None:
            local_rng = default_rng(seed + user_idx)
        else:
            local_rng = default_rng()

        items_positions = local_rng.choice(
            np.arange(pandas_df.shape[0]),
            size=min(k, pandas_df.shape[0]),
            p=pandas_df["probability"].values,
            replace=False,
        )

        return pd.DataFrame(
            {
                "user_idx": k * [user_idx],
                "item_idx": pandas_df["item_idx"].values[items_positions],
                "relevance": pandas_df["relevance"].values[items_positions],
            }
        )

    recs = pairs.groupby("user_idx").applyInPandas(grouped_map, REC_SCHEMA)

    return recs


def filter_cold(
    df: Optional[DataFrame],
    warm_df: DataFrame,
    col_name: str,
) -> Tuple[int, Optional[DataFrame]]:
    """
    Filter out new user/item ids absent in `warm_df`.
    Return number of new users/items and filtered dataframe.

    :param df: spark dataframe with columns ``[`col_name`, ...]``
    :param warm_df: spark dataframe with column ``[`col_name`]``,
        containing ids of `warm` users/items
    :param col_name: name of a column
    :return:  filtered spark dataframe columns ``[`col_name`, ...]``
    """
    if df is None:
        return 0, df

    num_cold = (
        df.select(col_name)
        .distinct()
        .join(warm_df, on=col_name, how="anti")
        .count()
    )

    if num_cold == 0:
        return 0, df

    return num_cold, df.join(
        warm_df.select(col_name), on=col_name, how="inner"
    )


def get_unique_entities(
    df: Union[Iterable, DataFrame],
    column: str,
) -> DataFrame:
    """
    Get unique values from ``df`` and put them into dataframe with column ``column``.
    :param df: spark dataframe with ``column`` or python iterable
    :param column: name of a column
    :return:  spark dataframe with column ``[`column`]``
    """
    spark = State().session
    if isinstance(df, DataFrame):
        unique = df.select(column).distinct()
    elif isinstance(df, collections.abc.Iterable):
        unique = spark.createDataFrame(
            data=pd.DataFrame(pd.unique(list(df)), columns=[column])
        )
    else:
        raise ValueError(f"Wrong type {type(df)}")
    return unique


def return_recs(
    recs: DataFrame, recs_file_path: Optional[str] = None
) -> Optional[DataFrame]:
    """
    Save dataframe `recs` to `recs_file_path` if presents otherwise cache
    and materialize the dataframe.

    :param recs: dataframe with recommendations
    :param recs_file_path: absolute path to save recommendations as a parquet file.
    :return: cached and materialized dataframe `recs` if `recs_file_path` is provided otherwise None
    """
    if recs_file_path is None:
        output = recs.cache()
        output.count()
        return output

    recs.write.parquet(path=recs_file_path, mode="overwrite")
    return None


def unionify(df: DataFrame, df_2: Optional[DataFrame] = None) -> DataFrame:
    if df_2 is not None:
        df = df.unionByName(df_2)
    return df


@contextmanager
def unpersist_after(dfs: Dict[str, Optional[DataFrame]]):
    yield

    for df in dfs.values():
        if df is not None:
            df.unpersist()


class AbleToSaveAndLoad(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: str, spark: Optional[SparkSession] = None):
        """
            load an instance of this class from saved state

            :return: an instance of the current class
        """

    @abstractmethod
    def save(self, path: str, overwrite: bool = False, spark: Optional[SparkSession] = None):
        """
            Saves the current instance
        """

    @staticmethod
    def _get_spark_session() -> SparkSession:
        return State().session

    @classmethod
    def _validate_classname(cls, classname: str):
        assert classname == cls.get_classname()

    @classmethod
    def get_classname(cls):
        return ".".join([cls.__module__, cls.__name__])


def prepare_dir(path):
    """
    Create empty `path` dir
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_class_by_name(classname: str) -> type:
    parts = classname.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def do_path_exists(path: str) -> bool:
    spark = State().session

    # due to the error: pyspark.sql.utils.IllegalArgumentException: Wrong FS: file:/...
    if path.startswith("file:/"):
        return os.path.exists(path)

    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    is_exists = fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))
    return is_exists


def list_folder(path: str) -> List[str]:
    """
        List files in a given directory
        :path: a directory to list files in
        :return: names of files from the given directory (not absolute names)
    """
    spark = State().session
    # if True:
    # # if path.startswith("file:/"):
    #
    #     files = [x for x in os.listdir(path)]
    #     logging.info("Files", files)
    #     return files
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    base_path = spark._jvm.org.apache.hadoop.fs.Path(path)

    if not fs.isDirectory(base_path):
        raise RuntimeError(f"The path is not directory. Cannot list it. The path: {path}")

    entries = fs.listStatus(base_path)
    files = [entry.getPath().getName() for entry in entries]
    return files


def save_transformer(
        transformer: AbleToSaveAndLoad,
        path: str,
        overwrite: bool = False):

    logger.info(f"Saving transformer on path: {path}")
    spark = State().session

    is_exists = do_path_exists(path)

    if is_exists and not overwrite:
        raise FileExistsError(f"Path '{path}' already exists. Mode is 'overwrite = False'.")
    elif is_exists:
        delete_folder(path)

    create_folder(path)

    spark.createDataFrame([{
        "classname": transformer.get_classname()
    }]).write.parquet(os.path.join(path, "metadata.parquet"))

    transformer.save(os.path.join(path, "transformer"), overwrite, spark=spark)
    logger.info(f"The transformer is saved on path {path}")


def load_transformer(path: str):
    spark = State().session
    metadata_row = spark.read.parquet(os.path.join(path, "metadata.parquet")).first().asDict()
    clazz = get_class_by_name(metadata_row["classname"])
    instance = clazz.load(os.path.join(path, "transformer"), spark)
    return instance


def save_picklable_to_parquet(obj: Any, path: str) -> None:
    """
    Function dumps object to disk or hdfs in parquet format.

    Args:
        obj: object to be saved
        path: path to dump
    """
    sc = SparkSession.getActiveSession().sparkContext
    # We can use `RDD.saveAsPickleFile`, but it has no "overwrite" parameter
    pickled_instance = pickle.dumps(obj)
    Record = collections.namedtuple("Record", ["data"])
    rdd = sc.parallelize([Record(pickled_instance)])
    instance_df = rdd.map(lambda rec: Record(bytearray(rec.data))).toDF()
    instance_df.write.mode("overwrite").parquet(path)


def load_pickled_from_parquet(path: str) -> Any:
    """
    Function loads object from disk or hdfs, what was dumped via `save_picklable_to_parquet` function.

    Args:
        path: source path

    Returns: unpickled object

    """
    spark = SparkSession.getActiveSession()
    df = spark.read.parquet(path)
    pickled_instance = df.rdd.map(lambda row: bytes(row.data)).first()
    return pickle.loads(pickled_instance)
