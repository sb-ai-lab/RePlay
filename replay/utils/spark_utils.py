import collections
import logging
import os
import pickle
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng

from .session_handler import State
from .types import PYSPARK_AVAILABLE, DataFrameLike, MissingImportType, NumType, PolarsDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.types as st
    from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
    from pyspark.sql import (
        Column,
        SparkSession,
        Window,
        functions as sf,
    )
    from pyspark.sql.column import _to_java_column, _to_seq
    from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
else:
    Column = MissingImportType


class PolarsConvertToSparkWarning(Warning):
    """
    Direct PolarsDataFrame to SparkDataFrame convertation warning.
    """


class SparkCollectToMasterWarning(Warning):  # pragma: no cover
    """
    Collect to master warning for Spark DataFrames.
    """


def spark_to_pandas(
    data: SparkDataFrame, allow_collect_to_master: bool = False, from_constructor: bool = True
) -> pd.DataFrame:  # pragma: no cover
    """
    Convert Spark DataFrame to Pandas DataFrame.

    :param data: Spark DataFrame.
    :param allow_collect_to_master: Flag allowing spark to make a collection to the master node, default: ``False``.

    :returns: Converted Pandas DataFrame.
    """
    warn_msg = "Spark Data Frame is collected to master node, this may lead to OOM exception for larger dataset. "
    if from_constructor:
        _msg = "To remove this warning set allow_collect_to_master=True in the recommender constructor."
    else:
        _msg = "To remove this warning set allow_collect_to_master=True."
    warn_msg += _msg
    if not allow_collect_to_master:
        warnings.warn(
            warn_msg,
            SparkCollectToMasterWarning,
        )
    return data.toPandas()


def convert2spark(data_frame: Optional[DataFrameLike]) -> Optional[SparkDataFrame]:
    """
    Converts Pandas DataFrame to Spark DataFrame

    :param data_frame: pandas DataFrame
    :return: converted data
    """
    if data_frame is None:
        return None
    if isinstance(data_frame, SparkDataFrame):
        return data_frame

    spark = State().session
    if isinstance(data_frame, PolarsDataFrame):
        warnings.warn(
            "Direct convertation PolarsDataFrame to SparkDataFrame currently is not supported, "
            "converting to pandas first",
            PolarsConvertToSparkWarning,
        )
        return spark.createDataFrame(data_frame.to_pandas())  # TODO: remove extra convertation to pandas
    return spark.createDataFrame(data_frame)


def get_top_k(
    dataframe: SparkDataFrame,
    partition_by_col: Column,
    order_by_col: List[Column],
    k: int,
) -> SparkDataFrame:
    """
    Return top ``k`` rows for each entity in ``partition_by_col`` ordered by
    ``order_by_col``.

    >>> from replay.utils.session_handler import State
    >>> spark = State().session
    >>> log = (
    ...    spark
    ...    .createDataFrame([(1, 2, 1.), (1, 3, 1.), (1, 4, 0.5), (2, 1, 1.)])
    ...    .toDF("user_id", "item_id", "relevance")
    ... )
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
            sf.row_number().over(Window.partitionBy(partition_by_col).orderBy(*order_by_col)),
        )
        .filter(sf.col("temp_rank") <= k)
        .drop("temp_rank")
    )


def get_top_k_recs(
    recs: SparkDataFrame,
    k: int,
    query_column: str = "user_idx",
    rating_column: str = "relevance",
) -> SparkDataFrame:
    """
    Get top k recommendations by `rating`.

    :param recs: recommendations DataFrame
        `[user_idx, item_idx, rating]`
    :param k: length of a recommendation list
    :param id_type: id or idx
    :return: top k recommendations `[user_id, item_id, rating]`
    """
    return get_top_k(
        dataframe=recs,
        partition_by_col=sf.col(query_column),
        order_by_col=[sf.col(rating_column).desc()],
        k=k,
    )


if PYSPARK_AVAILABLE:

    @sf.udf(returnType=st.DoubleType())
    def vector_dot(one: DenseVector, two: DenseVector) -> float:  # pragma: no cover
        """
        dot product of two column vectors

        >>> from replay.utils.session_handler import State
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
        StructType([StructField('dot', DoubleType(), True)])
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

    @sf.udf(returnType=VectorUDT())
    def vector_mult(one: Union[DenseVector, NumType], two: DenseVector) -> DenseVector:  # pragma: no cover
        """
        elementwise vector multiplication

        >>> from replay.utils.session_handler import State
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
        StructType([StructField('mult', VectorUDT(), True)])
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
    def array_mult(first: st.ArrayType, second: st.ArrayType):  # pragma: no cover
        """
        elementwise array multiplication

        >>> from replay.utils.session_handler import State
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
        StructType([StructField('mult', ArrayType(DoubleType(), True), True)])
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


def multiply_scala_udf(scalar, vector):
    """
    Multiplies a scalar by a vector

    :param scalar: column with scalars
    :param vector: column with vectors
    :return: column expression
    """
    sc = SparkSession.getActiveSession().sparkContext
    _f = sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.multiplyUDF()
    return Column(_f.apply(_to_seq(sc, [scalar, vector], _to_java_column)))


def get_log_info(log: SparkDataFrame, user_col="user_idx", item_col="item_idx") -> str:
    """
    Basic log statistics

    >>> from replay.utils.session_handler import State
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


def get_stats(log: SparkDataFrame, group_by: str = "user_id", target_column: str = "relevance") -> SparkDataFrame:
    """
    Calculate log statistics: min, max, mean, median ratings, number of ratings.
    >>> from replay.utils.session_handler import get_spark_session, State
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
        func(target_column).alias(str(name + "_" + target_column)) for name, func in agg_functions.items()
    ]
    agg_functions_list.append(sf.expr(f"percentile_approx({target_column}, 0.5)").alias("median_" + target_column))

    return log.groupBy(group_by).agg(*agg_functions_list)


def check_numeric(feature_table: SparkDataFrame) -> None:
    """
    Check if spark DataFrame columns are of NumericType
    :param feature_table: spark DataFrame
    """
    for column in feature_table.columns:
        if not isinstance(feature_table.schema[column].dataType, st.NumericType):
            msg = f"Column {column} has type {feature_table.schema[column].dataType}, that is not numeric."
            raise ValueError(msg)


def horizontal_explode(
    data_frame: SparkDataFrame,
    column_to_explode: str,
    prefix: str,
    other_columns: List[Column],
) -> SparkDataFrame:
    """
    Transform a column with an array of values into separate columns.
    Each array must contain the same amount of values.

    >>> from replay.utils.session_handler import State
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
        *[sf.element_at(column_to_explode, i + 1).alias(f"{prefix}_{i}") for i in range(num_columns)],
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
    base: SparkDataFrame,
    fill: SparkDataFrame,
    k: int,
    query_column: str = "user_idx",
    item_column: str = "item_idx",
    rating_column: str = "relevance",
) -> SparkDataFrame:
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
        return get_top_k_recs(fill, k, query_column=query_column, rating_column=rating_column)
    margin = 0.1
    min_in_base = base.agg({rating_column: "min"}).first()[0]
    max_in_fill = fill.agg({rating_column: "max"}).first()[0]
    diff = max_in_fill - min_in_base
    fill = fill.withColumnRenamed(rating_column, "relevance_fallback")
    if diff >= 0:
        fill = fill.withColumn("relevance_fallback", sf.col("relevance_fallback") - diff - margin)
    recs = base.join(fill, on=[query_column, item_column], how="full_outer")
    recs = recs.withColumn(rating_column, sf.coalesce(rating_column, "relevance_fallback")).select(
        query_column, item_column, rating_column
    )
    recs = get_top_k_recs(recs, k, query_column=query_column, rating_column=rating_column)
    return recs


def cache_if_exists(dataframe: Optional[SparkDataFrame]) -> Optional[SparkDataFrame]:
    """
    Cache a DataFrame
    :param dataframe: Spark DataFrame or None
    :return: DataFrame or None
    """
    if dataframe is not None:
        return dataframe.cache()
    return dataframe


def unpersist_if_exists(dataframe: Optional[SparkDataFrame]) -> None:
    """
    :param dataframe: DataFrame or None
    """
    if dataframe is not None and dataframe.is_cached:
        dataframe.unpersist()


def join_with_col_renaming(
    left: SparkDataFrame,
    right: SparkDataFrame,
    on_col_name: Union[str, List],
    how: str = "inner",
    suffix="join",
) -> SparkDataFrame:
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

    return (left.join(right, on=on_condition, how=how)).drop(*[f"{name}_{suffix}" for name in on_col_name])


def process_timestamp_column(
    dataframe: SparkDataFrame,
    column_name: str,
    date_format: Optional[str] = None,
) -> SparkDataFrame:
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
        msg = f"Column {column_name} not found"
        raise ValueError(msg)

    # no conversion needed
    if isinstance(dataframe.schema[column_name].dataType, st.TimestampType):
        return dataframe

    # unix timestamp
    if isinstance(dataframe.schema[column_name].dataType, st.NumericType):
        return dataframe.withColumn(column_name, sf.to_timestamp(sf.from_unixtime(sf.col(column_name))))

    # datetime in string format
    dataframe = dataframe.withColumn(
        column_name,
        sf.to_timestamp(sf.col(column_name), format=date_format),
    )
    return dataframe


if PYSPARK_AVAILABLE:

    @sf.udf(returnType=VectorUDT())
    def list_to_vector_udf(array: st.ArrayType) -> DenseVector:  # pragma: no cover
        """
        convert spark array to vector

        :param array: spark Array to convert
        :return:  spark DenseVector
        """
        return Vectors.dense(array)

    @sf.udf(returnType=st.FloatType())
    def vector_squared_distance(first: DenseVector, second: DenseVector) -> float:  # pragma: no cover
        """
        :param first: first vector
        :param second: second vector
        :returns: squared distance value
        """
        return float(first.squared_distance(second))

    @sf.udf(returnType=st.FloatType())
    def vector_euclidean_distance_similarity(first: DenseVector, second: DenseVector) -> float:  # pragma: no cover
        """
        :param first: first vector
        :param second: second vector
        :returns: 1/(1 + euclidean distance value)
        """
        return 1 / (1 + float(first.squared_distance(second)) ** 0.5)

    @sf.udf(returnType=st.FloatType())
    def cosine_similarity(first: DenseVector, second: DenseVector) -> float:  # pragma: no cover
        """
        :param first: first vector
        :param second: second vector
        :returns: cosine similarity value
        """
        num = first.dot(second)
        denom = first.dot(first) ** 0.5 * second.dot(second) ** 0.5
        return float(num / denom)


def cache_temp_view(df: SparkDataFrame, name: str) -> None:
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


def sample_top_k_recs(pairs: SparkDataFrame, k: int, seed: Optional[int] = None):
    """
    Sample k items for each user with probability proportional to the relevance score.

    Motivation: sometimes we have a pre-defined list of items for each user
    and could use `predict_pairs` method of RePlay models to score them.
    After that we could select top K most relevant items for each user
    with `replay.utils.spark_utils.get_top_k_recs` or sample them with
    probabilities proportional to their relevance score
    with `replay.utils.spark_utils.sample_top_k_recs` to get more diverse recommendations.

    :param pairs: spark dataframe with columns ``[user_idx, item_idx, relevance]``
    :param k: number of items for each user to return
    :param seed: random seed
    :return:  spark dataframe with columns ``[user_idx, item_idx, relevance]``
    """
    pairs = pairs.withColumn(
        "probability",
        sf.col("relevance") / sf.sum("relevance").over(Window.partitionBy("user_idx")),
    )

    def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        user_idx = pandas_df["user_idx"][0]

        local_rng = default_rng(seed + user_idx) if seed is not None else default_rng()

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

    rec_schema = StructType(
        [
            StructField("user_idx", IntegerType()),
            StructField("item_idx", IntegerType()),
            StructField("relevance", DoubleType()),
        ]
    )
    recs = pairs.groupby("user_idx").applyInPandas(grouped_map, rec_schema)

    return recs


def filter_cold(
    df: Optional[SparkDataFrame],
    warm_df: SparkDataFrame,
    col_name: str,
) -> Tuple[int, Optional[SparkDataFrame]]:
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

    num_cold = df.select(col_name).distinct().join(warm_df, on=col_name, how="anti").count()

    if num_cold == 0:
        return 0, df

    return num_cold, df.join(warm_df.select(col_name), on=col_name, how="inner")


def get_unique_entities(
    df: Union[Iterable, SparkDataFrame],
    column: str,
) -> SparkDataFrame:
    """
    Get unique values from ``df`` and put them into dataframe with column ``column``.
    :param df: spark dataframe with ``column`` or python iterable
    :param column: name of a column
    :return:  spark dataframe with column ``[`column`]``
    """
    spark = State().session
    if isinstance(df, SparkDataFrame):
        unique = df.select(column).distinct()
    elif isinstance(df, collections.abc.Iterable):
        unique = spark.createDataFrame(data=pd.DataFrame(pd.unique(list(df)), columns=[column]))
    else:
        msg = f"Wrong type {type(df)}"
        raise ValueError(msg)
    return unique


def return_recs(recs: SparkDataFrame, recs_file_path: Optional[str] = None) -> Optional[SparkDataFrame]:
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


def save_picklable_to_parquet(obj: Any, path: str) -> None:
    """
    Function dumps object to disk or hdfs in parquet format.

    :param obj: object to be saved
    :param path: path to dump
    :return:
    """

    sc = State().session.sparkContext
    # We can use `RDD.saveAsPickleFile`, but it has no "overwrite" parameter
    pickled_instance = pickle.dumps(obj)
    Record = collections.namedtuple("Record", ["data"])  # noqa: PYI024
    rdd = sc.parallelize([Record(pickled_instance)])
    instance_df = rdd.map(lambda rec: Record(bytearray(rec.data))).toDF()
    instance_df.write.mode("overwrite").parquet(path)


def load_pickled_from_parquet(path: str) -> Any:
    """
    Function loads object from disk or hdfs,
    what was dumped via `save_picklable_to_parquet` function.

    :param path: source path
    :return: unpickled object
    """
    spark = State().session
    df = spark.read.parquet(path)
    pickled_instance = df.rdd.map(lambda row: bytes(row.data)).first()
    return pickle.loads(pickled_instance)


def assert_omp_single_thread():
    """
    Check that OMP_NUM_THREADS is set to 1 and warn if not.

    PyTorch uses multithreading for cpu math operations via OpenMP library. Sometimes this
    leads to failures when OpenMP multithreading is mixed with multiprocessing.
    """
    omp_num_threads = os.environ.get("OMP_NUM_THREADS", None)
    if omp_num_threads != "1":
        msg = (
            f'Environment variable "OMP_NUM_THREADS" is set to "{omp_num_threads}". '
            f"Set it to 1 if the working process freezes."
        )
        logging.getLogger("replay").warning(msg)
