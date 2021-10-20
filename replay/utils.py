from typing import Any, List, Optional, Set, Union

import numpy as np
import pyspark.sql.types as st

from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
from pyspark.sql import Column, DataFrame, Window, functions as sf
from scipy.sparse import csr_matrix

from replay.constants import NumType, AnyDataFrame
from replay.session_handler import State

# pylint: disable=invalid-name


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


def get_top_k_recs(recs: DataFrame, k: int, id_type: str = "id") -> DataFrame:
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


def get_log_info(log: DataFrame) -> str:
    """
    Basic log statistics

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

    :param log: interaction log containing ``user_id`` and ``item_id``
    :returns: statistics string
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
    :param group_by: column to group data by, ``user_id`` или ``item_id``
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
    base: DataFrame, fill: DataFrame, k: int, id_type: str = "id"
) -> DataFrame:
    """
    Fill missing recommendations for users that have less than ``k`` recomended items.
    Score values for the fallback model may be decreased to preserve sorting.

    :param base: base recommendations that need to be completed
    :param fill: extra recommendations
    :param k: desired recommendation list lengths for each user
    :param id_type: id or idx
    :return: augmented recommendations
    """
    if fill is None:
        return base
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


def ugly_join(
    left: DataFrame,
    right: DataFrame,
    on_col_name: Union[str, List],
    how: str = "inner",
    suffix="join",
) -> DataFrame:
    """
    Ugly workaround for joining DataFrames derived form the same DataFrame
    https://issues.apache.org/jira/browse/SPARK-14948
    :param left: left-side dataframe
    :param right: right-side dataframe
    :param on_col_name: column name to join on
    :param how: join type
    :param suffix: suffix added to `on_col_name` value to name temporary column
    :return: join result
    """
    if isinstance(on_col_name, str):
        on_col_name = [on_col_name]

    on_condition = sf.lit(True)
    for name in on_col_name:
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
