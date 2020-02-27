"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import FloatType, StringType, TimestampType

from sponge_bob_magic import constants
from sponge_bob_magic.converter import convert
from sponge_bob_magic.utils import flat_list

CommonDataFrame = Union[DataFrame, pd.DataFrame]


class DataPreparator:
    """ Основной класс для считывания различных типов данных. """
    spark: SparkSession

    def __init__(self, spark: SparkSession):
        """
        Сохраняет спарк-сессию в качестве параметра.

        :param spark: инициализированная спарк-сессия
        """
        self.spark = spark

    def _read_data(self,
                   path: str,
                   format_type: str,
                   **kwargs) -> DataFrame:
        if format_type == "csv":
            dataframe = self.spark.read.csv(path, inferSchema=True, **kwargs)
        elif format_type == "parquet":
            dataframe = self.spark.read.parquet(path)
        elif format_type == "json":
            dataframe = self.spark.read.json(path, **kwargs)
        elif format_type == "table":
            dataframe = self.spark.read.table(path)
        else:
            raise ValueError(f"Invalid value of format_type='{format_type}'")

        return dataframe

    @staticmethod
    def _check_columns(given_columns: Set[str],
                       required_columns: Set[str],
                       optional_columns: Set[str]):
        if not required_columns.issubset(given_columns):
            raise ValueError(
                "В датафрейме нет обязательных колонок: "
                f"{required_columns.difference(given_columns)}")

        excess_columns = (given_columns
                          .difference(required_columns)
                          .difference(optional_columns))
        if len(excess_columns) > 0:
            raise ValueError("В 'columns_names' есть лишние колонки: "
                             f"{excess_columns}")

    @staticmethod
    def _check_dataframe(dataframe: DataFrame,
                         columns_names: Dict[str, Union[str, List[str]]]):
        # чекаем, что датафрейм не пустой
        if len(dataframe.head(1)) == 0:
            raise ValueError("Датафрейм пустой")

        # чекаем, что данные юзером колонки реально есть в датафрейме
        given_columns = set((flat_list(list(columns_names.values()))))
        dataframe_columns = set(dataframe.columns)
        if not given_columns.issubset(dataframe_columns):
            raise ValueError(
                "В columns_names в значениях есть колонки, "
                "которых нет в датафрейме: "
                f"{given_columns.difference(dataframe_columns)}")

        # чекаем на нуллы
        for column in given_columns:
            if dataframe.where(sf.col(column).isNull()).count() > 0:
                raise ValueError(f"В колонке '{column}' есть значения NULL")

    @staticmethod
    def _process_timestamp_column(
            dataframe: DataFrame,
            column_name: str,
            column: Column,
            date_format: Optional[str],
            default_value: str
    ):
        not_ts_types = ["timestamp", "string", None]
        if dict(dataframe.dtypes).get("timestamp", None) not in not_ts_types:
            # если в колонке лежат чиселки,
            # то это либо порядок записей, либо unix time

            # попробуем преобразовать unix time
            tmp_column = column_name + "tmp"
            dataframe = dataframe.withColumn(
                tmp_column,
                sf.to_timestamp(sf.from_unixtime(column))
            )

            # если не unix time, то в колонке будут все null
            is_null_column = (
                dataframe
                .select(
                    (sf.min(tmp_column).eqNullSafe(sf.max(tmp_column)))
                    .alias(tmp_column)
                )
                .collect()[0]
            )
            if is_null_column[tmp_column]:
                logging.warning(
                    "Колонка со временем не содержит unix time; "
                    "чиселки в этой колонке будут добавлены к "
                    "дефолтной дате")

                dataframe = (
                    dataframe
                    .withColumn("tmp",
                                sf.to_timestamp(
                                    sf.lit(default_value)))
                    .withColumn(column_name,
                                sf.to_timestamp(sf.expr(
                                    f"date_add(tmp, {column_name})")))
                    .drop("tmp", tmp_column)
                )
            else:
                dataframe = (
                    dataframe
                    .drop(column_name)
                    .withColumnRenamed(tmp_column, column_name)
                )
        else:
            dataframe = dataframe.withColumn(
                column_name,
                sf.to_timestamp(column, format=date_format)
            )
        return dataframe

    @staticmethod
    def _rename_columns(dataframe: DataFrame,
                        columns_names: Dict[str, Union[str, List[str]]],
                        default_schema: Dict[str, Tuple[Any, Any]],
                        date_format: Optional[str] = None):
        # колонки с фичами не будут переименованы, их надо просто селектить
        features_columns = columns_names.get("features", [])
        if not isinstance(features_columns, list):
            features_columns = [features_columns]
        # переименовываем колонки
        dataframe = dataframe.select([
            sf.col(column).alias(new_name)
            for new_name, column in columns_names.items()
            if new_name != "features"
        ] + features_columns)
        # добавляем необязательные дефолтные колонки, если их нет,
        # и задаем тип для тех колонок, что есть
        for column_name, (default_value,
                          default_type) in default_schema.items():
            if column_name not in dataframe .columns:
                column = sf.lit(default_value)
            else:
                column = sf.col(column_name)
            if column_name == "timestamp":
                dataframe = DataPreparator._process_timestamp_column(
                    dataframe, column_name, column, date_format,
                    default_value)
            else:
                dataframe = dataframe .withColumn(
                    column_name,
                    column.cast(default_type)
                )
        return dataframe


    def transform_log(self,
                      columns_names: Dict[str, Union[str, List[str]]],
                      log: Optional[CommonDataFrame] = None,
                      path: Optional[str] = None,
                      format_type: Optional[str] = None,
                      date_format: Optional[str] = None,
                      **kwargs) -> DataFrame:
        """
        Преобразовывает лог формата `format_type`
        в файле по пути `path` в спарк-датафрейм вида
        `[user_id, item_id, timestamp, context, relevance]`.

        :param columns_names: маппинг колонок, ключ - значения из списка
            `[user_id , item_id , timestamp , context , relevance]`;
            обязательными являются только `[user_id , item_id]`;
            значения - колонки в логе; в `timestamp` может быть числовая
            колонка, которая обозначает порядок записей,
            она будет преобразована в даты
        :param log: dataframe с логом
        :param path: путь к файлу с логом
        :param format_type: тип файла, принимает значения из списка
            `[csv , parquet , json , table]`
        :param date_format: формат даты, нужен, если формат даты особенный
        :param kwargs: дополнительные аргументы, которые передаются в функцию
            `spark.read.csv(path, **kwargs)`
        :return: спарк-датафрейм вида
            `[user_id , item_id , timestamp , context , relevance]`;
            колонки, не предоставленные в `columns_names`,
            заполянются дефолтными значениями
        """
        self._check_columns(set(columns_names.keys()),
                            required_columns={"user_id", "item_id"},
                            optional_columns={"timestamp", "context",
                                              "relevance"})

        if log:
            dataframe = convert(log)
        elif path and format_type:
            dataframe = self._read_data(path, format_type, **kwargs)
        else:
            raise ValueError("Один из параметров log, path должны быть отличным от None")

        self._check_dataframe(dataframe, columns_names)

        log_schema = {
            "timestamp": ("1999-05-01", TimestampType()),
            "context": (constants.DEFAULT_CONTEXT, StringType()),
            "relevance": (1.0, FloatType()),
            "user_id": (None, StringType()),
            "item_id": (None, StringType()),
        }
        dataframe = self._rename_columns(
            dataframe, columns_names,
            default_schema=log_schema,
            date_format=date_format
        ).cache()
        return dataframe

    def transform_features(self,
                           columns_names: Dict[str, Union[str, List[str]]],
                           log: Optional[CommonDataFrame] = None,
                           path: Optional[str] = None,
                           format_type: Optional[str] = None,
                           date_format: Optional[str] = None,
                           **kwargs) -> DataFrame:
        """
        Преобразовывает признаки формата `format_type`
        в файле по пути `path` в спарк-датафрейм вида
        `[user_id, timestamp, features]` или `[item_id, timestamp, features]`.

        :param columns_names: маппинг колонок, ключ - значения из списка
            `[user_id` / `item_id , timestamp , features]`;
            обязательными являются только
            `[user_id]` или `[item_id]` (должен быть один из них);
            если `features` нет в ключах,
            то призанками фичей явлются все оставшиеся колонки;
            в качестве `features` может подаваться как список, так и отдельное
            значение колонки (если признак один);
            значения - колонки в табличке признаков
        :param log: dataframe с логом
        :param path: путь к файлу с признаками
        :param format_type: тип файла, принимает значения из списка
            `[csv , parquet , json , table]`
        :param date_format: формат даты; нужен,
            если формат колонки `timestamp` особенный
        :param kwargs: дополнительные аргументы, которые передаются в функцию
            `spark.read.csv(path, **kwargs)`
        :return: спарк-датафрейм с колонками
            `[user_id / item_id , timestamp]` и колонки с признаками;
            колонка `timestamp`, если ее нет в `columns_names`,
            заполянются дефолтными значениями
        """
        if "user_id" in columns_names:
            required_columns = {"user_id"}
        elif "item_id" in columns_names:
            required_columns = {"item_id"}
        else:
            raise ValueError("В columns_names нет ни 'user_id', ни 'item_id'")

        self._check_columns(set(columns_names.keys()),
                            required_columns=required_columns,
                            optional_columns={"timestamp", "features"})
        if log:
            dataframe = convert(log)
        elif path and format_type:
            dataframe = self._read_data(path, format_type, **kwargs)
        else:
            raise ValueError("Один из параметров log, path должны быть отличным от None")
        # если фичей нет в данных юзером колонках, вставляем все оставшиеся
        # нужно, чтобы проверить, что там нет нуллов
        if "features" not in columns_names:
            given_columns = flat_list(list(columns_names.values()))
            dataframe_columns = dataframe.columns
            feature_columns = list(
                set(dataframe_columns).difference(given_columns)
            )
            if len(feature_columns) == 0:
                raise ValueError("В датафрейме нет колонок с фичами")
            features_dict = {"features": feature_columns}
        else:
            features_dict = dict()
        self._check_dataframe(dataframe, {**columns_names, **features_dict})
        features_schema = {
            "timestamp": ("1999-05-01", TimestampType()),
            ("user_id" if "user_id" in columns_names else "item_id"):
                (None, StringType()),
        }
        dataframe = self._rename_columns(
            dataframe, columns_names,
            default_schema=features_schema,
            date_format=date_format).cache()
        return dataframe
