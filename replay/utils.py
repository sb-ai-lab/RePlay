from typing import Any, List, Optional, Set, Union

import numpy as np
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql import Column, DataFrame, Window, functions as sf
from pyspark.sql.types import ArrayType, DoubleType, NumericType
from scipy.sparse import csr_matrix

from replay.constants import NumType, AnyDataFrame
from replay.session_handler import State

# pylint: disable=invalid-name


def convert2spark(data_frame: Optional[AnyDataFrame]) -> Optional[DataFrame]:
    """
    Обеспечивает конвертацию данных в спарк и обратно.

    :param data_frame: данные в формате датафрейма pandas или spark,
        либо объект датасета, в котором лежат датафреймы поддерживаемых форматов.
    :return: преобразованные данные, если на вход был подан датафрейм.
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


def get_top_k_recs(recs: DataFrame, k: int, id_type: str = "id") -> DataFrame:
    """
    Выбирает из рекомендаций топ-k штук на основе `relevance`.

    :param recs: рекомендации, спарк-датафрейм с колонками
        `[user_id, item_id, relevance]`
    :param k: число рекомендаций для каждого пользователя
    :param id_type: использовать id или idx в колонках
    :return: топ-k рекомендации, спарк-датафрейм с колонками
        `[user_id, item_id, relevance]`
    """
    window = Window.partitionBy(recs["user_" + id_type]).orderBy(
        recs["relevance"].desc()
    )
    return (
        recs.withColumn("rank", sf.row_number().over(window))
        .filter(sf.col("rank") <= k)
        .drop("rank")
    )


@sf.udf(returnType=DoubleType())  # type: ignore
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


@sf.udf(returnType=VectorUDT())  # type: ignore
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


@sf.udf(returnType=ArrayType(DoubleType()))
def array_mult(first: Column, second: Column):
    """
    Покоординатное произведение двух столбцов типа array.

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

    :param first: первый множитель
    :param second: второй множитель
    :returns: вектор с результатом покоординатного умножения
    """

    return [first[i] * second[i] for i in range(len(first))]


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


def get_stats(
    log: DataFrame, group_by: str = "user_id", target_column: str = "relevance"
) -> DataFrame:
    """
    Подсчет статистик (минимальная, максимальная, средняя, медианая оценки, число оценок) по логу взаимодействия.
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
    >>> get_stats(test_df, group_by='item_id', target_column='rel').show()
    +-------+--------+-------+-------+---------+----------+
    |item_id|mean_rel|max_rel|min_rel|count_rel|median_rel|
    +-------+--------+-------+-------+---------+----------+
    |      2|     1.0|      1|      1|        1|         1|
    |      3|     2.5|      3|      2|        2|         2|
    |      1|     2.0|      2|      2|        1|         2|
    +-------+--------+-------+-------+---------+----------+

    :param log: spark DataFrame с колонками ``user_id``, ``item_id`` и ``relevance``
    :param group_by: колонка для группировки, ``user_id`` или ``item_id``
    :param target_column: колонка с оценками взаимодействия, ``relevance``
    :return: spark DataFrame со статистиками взаимодействия по пользователям|объектам
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
        sf.expr("percentile_approx({}, 0.5)".format(target_column)).alias(
            "median_" + target_column
        )
    )

    return log.groupBy(group_by).agg(*agg_functions_list)


def check_numeric(feature_table: DataFrame) -> None:
    """
    Проверяет, что столбцы spark DataFrame feature_table принадлежат к типу NumericType
    :param feature_table: spark DataFrame, типы столбцов которого нужно проверить
    """
    for column in feature_table.columns:
        if not isinstance(feature_table.schema[column].dataType, NumericType):
            raise ValueError(
                "Столбец {} имеет неверный тип {}, столбец должен иметь числовой тип.".format(
                    column, feature_table.schema[column].dataType
                )
            )


def to_csr(
    log: DataFrame,
    user_count: Optional[int] = None,
    item_count: Optional[int] = None,
) -> csr_matrix:
    """
    Конвертирует лог в csr матрицу user-item.

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 2], "relevance": [1, 2]})
    >>> data_frame = convert2spark(data_frame)
    >>> m = to_csr(data_frame)
    >>> m.toarray()
    array([[1, 0, 0],
           [0, 0, 2]])

    :param log: spark DataFrame с колонками ``user_id``, ``item_id`` и
    ``relevance``
    :param user_count: количество строк в результирующей матрице (если пусто,
    то вычисляется по логу)
    :param item_count: количество столбцов в результирующей матрице (если
    пусто, то вычисляется по логу)
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
    >>> horizontal_explode(input_data, "array_col", "element", [sf.col("id_col")]).show()
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
            sf.element_at(column_to_explode, i + 1).alias(f"{prefix}_{i}")
            for i in range(num_columns)
        ],
    )


def join_or_return(first, second, on, how):
    """
    Обертка над join для удобного join-а датафреймов, например лога с признаками.
    Если датафрейм second есть, будет выполнен join, иначе возвращен first.

    :param first: spark-dataframe
    :param second: spark-dataframe
    :param on: имя столбца, по которому выполняется join
    :param how: тип join
    :return: spark-dataframe
    """
    if second is None:
        return first
    return first.join(second, on=on, how=how)


def fallback(
    base: DataFrame, fill: DataFrame, k: int, id_type: str = "id"
) -> DataFrame:
    """Подмешивает к основным рекомендациям запасные
    для пользователей, у которых количество рекомендаций меньше ``k``.
    Скор дополнительной модели может быть уменьшен,
    чтобы первыми были основные рекомендации.

    :param base: основные рекомендации
    :param fill: запасные рекомендации
    :param k: сколько должно быть для каждого пользователя
    :param id_type: использовать id или idx в колонках
    :return: дополненные рекомендации
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


# pylint: disable=too-many-locals
def get_first_level_model_features(
    model: DataFrame,
    pairs: DataFrame,
    user_features: Optional[DataFrame] = None,
    item_features: Optional[DataFrame] = None,
    add_factors_mult: bool = True,
) -> DataFrame:
    """
    Добавление векторов пользователей и объектов из модели replay.
    Если модель может вернуть и вектора пользователей, и вектора объектов,
    можно дополнительно посчитать покомпонентное произведение. Настраивается параметром add_factors_mult.
    Если модель не может вернуть вектора для части пользователей/объектов, для них возвращаются нулевые вектора.

    :param model: обученная модель replay, возвращающая вектора пользователей/объектов
    :param pairs: пары пользователь/объект для которых нужно вернуть вектора
        spark-датафрейм с колонками `[user_id/user_idx, item_id/item_id]`
    :param user_features: датафрейм, содержащий признаки пользователей,
        spark-датафрейм с колонками `[user_id/user_idx, feature_1, ....]`
    :param item_features: spark-датафрейм, содержащий признаки объектов
        spark-датафрейм с колонками `[item_id/item_idx, feature_1, ....]`
    :param add_factors_mult: добавить ли в качестве признаков результат покомпонентного умножения векторов
    :return: spark-датафрейм, содержащий компоненты векторов в качестве отдельных колонок
    """
    if "user_id" in pairs.columns:
        func_name = "_get_features_wrap"
        suffix = "id"
    else:
        func_name = "_get_features"
        suffix = "idx"

    users = pairs.select("user_{}".format(suffix)).distinct()
    items = pairs.select("item_{}".format(suffix)).distinct()
    user_factors, user_vector_len = getattr(model, func_name)(
        users, user_features
    )
    item_factors, item_vector_len = getattr(model, func_name)(
        items, item_features
    )

    pairs_with_features = join_or_return(
        pairs, user_factors, how="left", on="user_{}".format(suffix)
    )
    pairs_with_features = join_or_return(
        pairs_with_features,
        item_factors,
        how="left",
        on="item_{}".format(suffix),
    )

    factors_to_explode = []
    if user_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "user_factors",
            sf.coalesce(
                sf.col("user_factors"),
                sf.array([sf.lit(0.0)] * user_vector_len),
            ),
        )
        factors_to_explode.append(("user_factors", "uf"))

    if item_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "item_factors",
            sf.coalesce(
                sf.col("item_factors"),
                sf.array([sf.lit(0.0)] * item_vector_len),
            ),
        )
        factors_to_explode.append(("item_factors", "if"))

    if model.__str__() == "LightFMWrap":
        pairs_with_features.fillna({"user_bias": 0, "item_bias": 0})

    if (
        add_factors_mult
        and user_factors is not None
        and item_factors is not None
    ):
        pairs_with_features = pairs_with_features.withColumn(
            "factors_mult",
            array_mult(sf.col("item_factors"), sf.col("user_factors")),
        )
        factors_to_explode.append(("factors_mult", "fm"))

    for col_name, prefix in factors_to_explode:
        col_set = set(pairs_with_features.columns)
        col_set.remove(col_name)
        pairs_with_features = horizontal_explode(
            data_frame=pairs_with_features,
            column_to_explode=col_name,
            other_columns=[sf.col(column) for column in sorted(list(col_set))],
            prefix=prefix,
        )

    return pairs_with_features


def cache_if_exists(dataframe: Optional[DataFrame]) -> Optional[DataFrame]:
    """
    Возвращает кэшированный датафрейм
    :param dataframe: spark-датафрейм или None
    :return: кэшированный spark-датафрейм или None
    """
    if dataframe is not None:
        return dataframe.cache()
    return dataframe


def unpersist_if_exists(dataframe: Optional[DataFrame]) -> None:
    """
    Применяет unpersist к spark-датафрейму
    :param dataframe: кэшированный spark-датафрейм или None
    """
    if dataframe is not None:
        dataframe.unpersist()
