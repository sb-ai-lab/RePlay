"""
Для получения рекомендаций данные необходимо предварительно обработать.
Они должны соответствовать определенной структуре. Перед тем как обучить модель
необходимо воспользоваться классом ``DataPreparator``. Данные пользователя могут храниться в файле
либо содержаться внутри объекта ``pandas.DataFrame`` или ``spark.DataFrame``.

"""
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import FloatType, StringType, TimestampType

from sponge_bob_magic import constants
from sponge_bob_magic.converter import convert
from sponge_bob_magic.session_handler import State

CommonDataFrame = Union[DataFrame, pd.DataFrame]


class DataPreparator:
    """ Класс для преобразования различных типов данных.
    Для преобразования данных необходимо иницализировать объект класс
    ``DataPreparator`` и вызвать метод ``transform``. В случае, если в нем указан мапинг
    столбцов  "user_id" и "item_id", то считается, что пользователь передал таблицу с логом,
    если же в мапинге указан только один из столбцов "user_id"/"item_id", то передана
    таблица с признаками пользователей/объектов соответственно.

    Примеры использования:

    Загрузка таблицы с логом (слобцы "user_id" и "item_id" обязательны).

    >>> import pandas as pd
    >>> from sponge_bob_magic.data_preparator import DataPreparator
    >>>
    >>> log = pd.DataFrame({"user_id": [2, 2, 2, 1],
    ...                     "item_id": [1, 2, 3, 3],
    ...                     "relevance": [5, 5, 5, 5]}
    ...                    )
    >>> dp = DataPreparator()
    >>> correct_log = dp.transform(data=log,
    ...                            columns_names={"user_id": "user_id",
    ...                                           "item_id": "item_id",
    ...                                           "relevance": "relevance"}
    ...                             )
    >>> correct_log.show(2)
    +-------+-------+---------+-------------------+
    |user_id|item_id|relevance|          timestamp|
    +-------+-------+---------+-------------------+
    |      2|      1|      5.0|1999-05-01 00:00:00|
    |      2|      2|      5.0|1999-05-01 00:00:00|
    +-------+-------+---------+-------------------+
    only showing top 2 rows
    <BLANKLINE>


    Загрузка таблицы с признакми пользователя (обязателен один из столбцов "user_id" и "item_id").

    >>> import pandas as pd
    >>> from sponge_bob_magic.data_preparator import DataPreparator
    >>>
    >>> log = pd.DataFrame({"user": ["user1", "user1", "user2"],
    ...                     "f0": ["feature1","feature2","feature1"],
    ...                     "f1": ["left","left","center"],
    ...                     "ts": ["2019-01-01","2019-01-01","2019-01-01"]}
    ...             )
    >>> dp = DataPreparator()
    >>> correct_log = dp.transform(data=log,
    ...                            columns_names={"user_id": "user",
    ...                                           "timestamp": "ts"},
    ...                            features_columns=["f0"]
    ...                             )
    >>> correct_log.show(3)
    +-------+-------------------+--------+
    |user_id|          timestamp|      f0|
    +-------+-------------------+--------+
    |  user1|2019-01-01 00:00:00|feature1|
    |  user1|2019-01-01 00:00:00|feature2|
    |  user2|2019-01-01 00:00:00|feature1|
    +-------+-------------------+--------+
    <BLANKLINE>

    Загрузка таблицы с признакми пользователя без явной передачи списка признаков.
    В случае если параметр features_columns не задан, признаками считаются все остальные столбцы.

    >>> import pandas as pd
    >>> from sponge_bob_magic.data_preparator import DataPreparator
    >>>
    >>> log = pd.DataFrame({"user": ["user1", "user1", "user2"],
    ...                     "f0": ["feature1","feature2","feature1"],
    ...                     "f1": ["left","left","center"],
    ...                     "ts": ["2019-01-01","2019-01-01","2019-01-01"]}
    ...             )
    >>> dp = DataPreparator()
    >>> correct_log = dp.transform(data=log,
    ...                            columns_names={"user_id": "user",
    ...                                           "timestamp": "ts"}
    ...                             )
    >>> correct_log.show(3)
    +-------+-------------------+--------+------+
    |user_id|          timestamp|      f0|    f1|
    +-------+-------------------+--------+------+
    |  user1|2019-01-01 00:00:00|feature1|  left|
    |  user1|2019-01-01 00:00:00|feature2|  left|
    |  user2|2019-01-01 00:00:00|feature1|center|
    +-------+-------------------+--------+------+
    <BLANKLINE>
    """
    def _read_data(self,
                   path: str,
                   format_type: str,
                   **kwargs) -> DataFrame:
        spark = State().session
        if format_type == "csv":
            dataframe = spark.read.csv(path, inferSchema=True, **kwargs)
        elif format_type == "parquet":
            dataframe = spark.read.parquet(path)
        elif format_type == "json":
            dataframe = spark.read.json(path, **kwargs)
        elif format_type == "table":
            dataframe = spark.read.table(path)
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
        if excess_columns:
            raise ValueError("В 'columns_names' есть лишние колонки: "
                             f"{excess_columns}")

    @staticmethod
    def _check_dataframe(dataframe: DataFrame,
                         columns_names: Dict[str, str]):
        # чекаем, что датафрейм не пустой
        if not dataframe.head(1):
            raise ValueError("Датафрейм пустой")

        # чекаем, что данные юзером колонки реально есть в датафрейме
        given_columns = set(columns_names.values())
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
                        columns_names: Dict[str, str],
                        features_columns: List[str],
                        default_schema: Dict[str, Tuple[Any, Any]],
                        date_format: Optional[str] = None):
        # переименовываем колонки
        dataframe = dataframe.select([
            sf.col(column).alias(new_name)
            for new_name, column in columns_names.items()
        ] + features_columns)
        # добавляем необязательные дефолтные колонки, если они есть,
        # и задаем тип для тех колонок, что есть
        for column_name, (default_value,
                          default_type) in default_schema.items():
            if column_name not in dataframe.columns:
                column = sf.lit(default_value)
            else:
                column = sf.col(column_name)
            if column_name == "timestamp":
                dataframe = DataPreparator._process_timestamp_column(
                    dataframe, column_name, column, date_format,
                    default_value)
            else:
                dataframe = dataframe.withColumn(
                    column_name,
                    column.cast(default_type)
                )
        return dataframe

    def transform(self,
                  columns_names: Dict[str, str],
                  data: Optional[CommonDataFrame] = None,
                  path: Optional[str] = None,
                  format_type: Optional[str] = None,
                  date_format: Optional[str] = None,
                  features_columns: Optional[Union[str, Iterable[str]]] = None,
                  **kwargs) -> DataFrame:
        """
        Преобразовывает лог, либо признаки пользователей или объектов
        в спарк-датафрейм вида
        ``[user_id, timestamp, *features]`` или ``[item_id, timestamp, *features]``
        или ``[user_id, user_id, timestamp, relevance]``.
        На вход необходимо передать либо файл формата ``format_type``
        по пути ``path``, либо ``pandas.DataFrame`` или ``spark.DataFrame``.

        :param columns_names: маппинг колонок, ключ-значение из списка
            ``[user_id / item_id , timestamp , *columns]``;
            обязательными являются только ``[user_id]`` или ``[item_id]``
            (должен быть хотя бы один из них);
            В зависимости от маппинга определяется какого типа таблица передана.

            - Если присутствуют оба столбца ``[user_id, item_id]``, то передана таблица с логом
            - Если присутствует только ``[user_id]``, то передана таблица с признаками пользователей
            - Если присутствует только ``[item_id]``, то передана таблица с признаками объектов

        :param data: dataframe с логом
        :param path: путь к файлу с признаками
        :param format_type: тип файла, принимает значения из списка
            ``[csv , parquet , json , table]``
        :param date_format: формат даты; нужен,
            если формат колонки ``timestamp`` особенный
        :param features_columns: столбец либо список столбцов, в которых хранятся
            признаки пользователей/объектов. Если ``features`` пуст и при этом
            передается таблица с признаками пользователей или объектов,
            то признаками фичей явлются все оставшиеся колонки;
            в качестве ``features`` может подаваться как список, так и отдельное
            значение колонки (если признак один);
            значения - колонки в табличке признаков
        :param kwargs: дополнительные аргументы, которые передаются в функцию
            ``spark.read.csv(path, **kwargs)``
        :return: спарк-датафрейм с колонками
            ``[user_id / item_id , timestamp]`` и прочие колонки из ``columns_names``;
            колонки, не предоставленные в ``columns_names``,
            заполянются дефолтными значениями
        """
        if data is not None:
            dataframe = convert(data)
        elif path and format_type:
            dataframe = self._read_data(path, format_type, **kwargs)
        else:
            raise ValueError("Один из параметров data, path должен быть отличным от None")

        if "user_id" in columns_names and "item_id" in columns_names:
            required_columns = {"user_id": (None, StringType()),
                                "item_id": (None, StringType())}
            optional_columns = {"timestamp": ("1999-05-01", TimestampType()),
                                "relevance": (1.0, FloatType())}
            if features_columns is None:
                features_columns = []
            else:
                raise ValueError("В данной таблице features не используются")
        else:
            optional_columns = {"timestamp": ("1999-05-01", TimestampType())}
            if "user_id" in columns_names:
                required_columns = {"user_id": (None, StringType())}
            elif "item_id" in columns_names:
                required_columns = {"item_id": (None, StringType())}
            else:
                raise ValueError("В columns_names нет ни 'user_id', ни 'item_id'")

            # если фичей нет в данных пользователем колонках, вставляем все оставшиеся
            # нужно, чтобы проверить, что там нет нуллов
            if features_columns is None:
                given_columns = set(columns_names.values())
                dataframe_columns = set(dataframe.columns)
                features_columns = sorted(list(dataframe_columns.
                                               difference(given_columns)))
                if not features_columns:
                    raise ValueError("В датафрейме нет колонок с фичами")

            else:
                if isinstance(features_columns, str):
                    features_columns = [features_columns]
                else:
                    features_columns = list(features_columns)

        self._check_columns(set(columns_names.keys()),
                            required_columns=set(required_columns),
                            optional_columns=set(optional_columns))

        self._check_dataframe(dataframe, columns_names)

        dataframe2 = self._rename_columns(
            dataframe, columns_names, features_columns,
            default_schema={**required_columns, **optional_columns},
            date_format=date_format).cache()
        return dataframe2
