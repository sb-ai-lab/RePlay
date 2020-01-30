"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from math import floor
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import psutil
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as sf


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


def write_read_dataframe(
        spark: SparkSession,
        dataframe: DataFrame,
        path: Optional[str],
        to_overwrite_files: bool = True
) -> DataFrame:
    """
    Записывает спарк-датафрейм на диск и считывает его обратно и возвращает.
    Если путь равен None, то возвращается спарк-датафрейм, поданный на вход.

    :param to_overwrite_files: флажок, если True, то перезаписывает файл,
        если он существует; иначе - поднимается исключение
    :param spark: инициализированная спарк-сессия
    :param dataframe: спарк-датафрейм
    :param path: путь, по которому происходит записаь датафрейма
    :return: оригинальный датафрейм; если `path` не пустой,
        то lineage датафрейма обнуляется
    """
    if path is not None:
        (dataframe
         .write
         .mode("overwrite" if to_overwrite_files else "error")
         .parquet(path))
        dataframe = spark.read.parquet(path)
    return dataframe


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
        `[user_id , item_id , context , relevance]`
    :param k: число рекомендаций для каждого юзера
    :return: топ-k рекомендации, спарк-датафрейм с колонками
        `[user_id , item_id , context , relevance]`
    """
    window = (Window
              .partitionBy(recs["user_id"])
              .orderBy(recs["relevance"].desc()))
    return (recs
            .withColumn("rank",
                        sf.row_number().over(window))
            .filter(sf.col("rank") <= k)
            .drop("rank"))


def get_spark_session(spark_memory: Optional[int] = None) -> SparkSession:
    """
    инициализирует и возращает SparkSession с "годными" параметрами по
    умолчанию (для пользователей, которые не хотят сами настраивать Spark)

    :param spark_memory: количество гигабайт оперативной памяти, которую нужно выделить под Spark;
        если не задано, выделяется половина всей доступной памяти
    """
    if spark_memory is None:
        spark_memory = floor(psutil.virtual_memory().total / 1024 ** 3 / 2)
    spark_cores = "*"
    user_home = os.environ["HOME"]
    spark = (
        SparkSession
        .builder
        .config("spark.driver.memory", f"{spark_memory}g")
        .config("spark.local.dir", os.path.join(user_home, "tmp"))
        .master(f"local[{spark_cores}]")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.WARN)
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    return spark
