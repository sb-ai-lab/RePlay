"""
Содержит классы  ``DataPreparator`` и ``CatFeaturesTransformer``.
``DataPreparator`` используется для приведения лога и признаков в формат библиотеки.
Для получения рекомендаций данные необходимо предварительно обработать.
Они должны соответствовать определенной структуре. Перед тем как обучить модель
необходимо воспользоваться классом ``DataPreparator``. Данные пользователя могут храниться в файле
либо содержаться внутри объекта ``pandas.DataFrame`` или ``spark.DataFrame``.

`CatFeaturesTransformer`` позволяет удобным образом трансформировать
категориальные признаки с помощью one-hot encoding.
"""
import logging
import string
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    DoubleType,
    StringType,
    TimestampType,
    NumericType,
)

from replay.constants import AnyDataFrame
from replay.utils import convert2spark
from replay.session_handler import State


# pylint: disable=too-few-public-methods
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
    >>> from replay.data_preparator import DataPreparator
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
    >>> from replay.data_preparator import DataPreparator
    >>>
    >>> log = pd.DataFrame({"user": ["user1", "user1", "user2"],
    ...                     "f0": ["feature1","feature2","feature1"],
    ...                     "f1": ["left","left","center"],
    ...                     "ts": ["2019-01-01","2019-01-01","2019-01-01"]}
    ...             )
    >>> dp = DataPreparator()
    >>> correct_log = dp.transform(data=log,
    ...                            columns_names={"user_id": "user"},
    ...                            features_columns=["f0"]
    ...                             )
    >>> correct_log.show(3)
    +-------+--------+
    |user_id|      f0|
    +-------+--------+
    |  user1|feature1|
    |  user1|feature2|
    |  user2|feature1|
    +-------+--------+
    <BLANKLINE>

    Загрузка таблицы с признаками пользователя без явной передачи списка признаков.
    В случае если параметр features_columns не задан, признаками считаются все остальные столбцы.

    >>> import pandas as pd
    >>> from replay.data_preparator import DataPreparator
    >>>
    >>> log = pd.DataFrame({"user": ["user1", "user1", "user2"],
    ...                     "f0": ["feature1","feature2","feature1"],
    ...                     "f1": ["left","left","center"],
    ...                     "ts": ["2019-01-01","2019-01-01","2019-01-01"]}
    ...             )
    >>> dp = DataPreparator()
    >>> correct_log = dp.transform(data=log,
    ...                            columns_names={"user_id": "user"}
    ...                             )
    >>> correct_log.show(3)
    +-------+--------+------+----------+
    |user_id|      f0|    f1|        ts|
    +-------+--------+------+----------+
    |  user1|feature1|  left|2019-01-01|
    |  user1|feature2|  left|2019-01-01|
    |  user2|feature1|center|2019-01-01|
    +-------+--------+------+----------+
    <BLANKLINE>
    """

    @staticmethod
    def _read_data(path: str, format_type: str, **kwargs) -> DataFrame:
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
    def _check_columns(
        given_columns: Set[str],
        required_columns: Set[str],
        optional_columns: Set[str],
    ):
        if not required_columns.issubset(given_columns):
            raise ValueError(
                "В датафрейме нет обязательных колонок: "
                f"{required_columns.difference(given_columns)}"
            )

        excess_columns = given_columns.difference(required_columns).difference(
            optional_columns
        )
        if excess_columns:
            raise ValueError(
                "В 'columns_names' есть лишние колонки: " f"{excess_columns}"
            )

    @staticmethod
    def _check_dataframe(
        dataframe: DataFrame,
        columns_names: Dict[str, str],
        feature_columns: List[str],
    ):
        # чекаем, что датафрейм не пустой
        if not dataframe.head(1):
            raise ValueError("Датафрейм пустой")

        # чекаем, что данные юзером колонки реально есть в датафрейме
        columns_to_check = {*columns_names.values(), *feature_columns}
        dataframe_columns = set(dataframe.columns)
        if not columns_to_check.issubset(dataframe_columns):
            raise ValueError(
                "В columns_names в значениях есть колонки, "
                "которых нет в датафрейме: "
                f"{columns_to_check.difference(dataframe_columns)}"
            )

        # чекаем на нуллы только столбцы из columns_names
        for column in columns_names.values():
            if dataframe.where(sf.col(column).isNull()).count() > 0:
                raise ValueError(f"В колонке '{column}' есть значения NULL")

    @staticmethod
    def _process_timestamp_column(
        dataframe: DataFrame,
        column_name: str,
        column: Column,
        date_format: Optional[str],
        default_value: str,
    ):
        not_ts_types = ["timestamp", "string", None]
        if dict(dataframe.dtypes).get("timestamp", None) not in not_ts_types:
            # если в колонке лежат чиселки,
            # то это либо порядок записей, либо unix time

            # попробуем преобразовать unix time
            tmp_column = column_name + "tmp"
            dataframe = dataframe.withColumn(
                tmp_column, sf.to_timestamp(sf.from_unixtime(column))
            )

            # если не unix time, то в колонке будут все null
            is_null_column = dataframe.select(
                (sf.min(tmp_column).eqNullSafe(sf.max(tmp_column))).alias(
                    tmp_column
                )
            ).collect()[0]
            if is_null_column[tmp_column]:
                logger = logging.getLogger("replay")
                logger.warning(
                    "Колонка со временем не содержит unix time; "
                    "чиселки в этой колонке будут добавлены к "
                    "дефолтной дате"
                )

                dataframe = (
                    dataframe.withColumn(
                        "tmp", sf.to_timestamp(sf.lit(default_value))
                    )
                    .withColumn(
                        column_name,
                        sf.to_timestamp(
                            sf.expr(f"date_add(tmp, {column_name})")
                        ),
                    )
                    .drop("tmp", tmp_column)
                )
            else:
                dataframe = dataframe.drop(column_name).withColumnRenamed(
                    tmp_column, column_name
                )
        else:
            dataframe = dataframe.withColumn(
                column_name,
                sf.to_timestamp(column, format=date_format),  # type: ignore
            )
        return dataframe

    @staticmethod
    def _rename_columns(
        dataframe: DataFrame,
        columns_names: Dict[str, str],
        features_columns: List[str],
        default_schema: Dict[str, Tuple[Any, Any]],
        date_format: Optional[str] = None,
    ):
        # переименовываем колонки
        dataframe = dataframe.select(
            [
                sf.col(column).alias(new_name)
                for new_name, column in columns_names.items()
            ]
            + [sf.col(column) for column in features_columns]
        )
        # добавляем необязательные дефолтные колонки, если они есть,
        # и задаем тип для тех колонок, что есть
        for (
            column_name,
            (default_value, default_type),
        ) in default_schema.items():
            if column_name not in dataframe.columns:
                column = sf.lit(default_value)
            else:
                column = sf.col(column_name)
            if column_name == "timestamp":
                dataframe = DataPreparator._process_timestamp_column(
                    dataframe, column_name, column, date_format, default_value
                )
            else:
                dataframe = dataframe.withColumn(
                    column_name, column.cast(default_type)
                )
        return dataframe

    # pylint: disable=too-many-arguments
    def transform(
        self,
        columns_names: Dict[str, str],
        data: Optional[AnyDataFrame] = None,
        path: Optional[str] = None,
        format_type: Optional[str] = None,
        date_format: Optional[str] = None,
        features_columns: Optional[Union[str, Iterable[str]]] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Преобразовывает лог, либо признаки пользователей или объектов
        в спарк-датафрейм вида
        ``[user_id, *features]`` или ``[item_id, *features]``
        или ``[user_id, user_id, timestamp, relevance]``.
        На вход необходимо передать либо файл формата ``format_type``
        по пути ``path``, либо ``pandas.DataFrame`` или ``spark.DataFrame``.
        :param columns_names: словарь "стандартное имя столбца: имя столбца в dataframe" для лога обязательно задать
        соответствие для столбцов ``user_id`` и ``item_id``, опционально можно указать соответствия для столбцов
        ``timestamp`` (время взаимодействия) и ``relevance`` (релевантность, оценка взаимодействия). Если
        соответствие для данных столбцов не указаны, они будут созданы автоматически с дефолтными значениями.

            для таблиц признаков пользователей и объектов необходимо задать соответствие для ``user_id``
             или ``item_id``.

            В зависимости от маппинга определяется какого типа таблица передана.
            - Если присутствуют оба столбца ``[user_id, item_id]``, то передана таблица с логом
            - Если присутствует только ``[user_id]``, то передана таблица с признаками пользователей
            - Если присутствует только ``[item_id]``, то передана таблица с признаками объектов

        :param data: dataframe с логом
        :param path: путь к файлу с данными
        :param format_type: тип файла, принимает значения из списка
            ``[csv , parquet , json , table]``
        :param date_format: формат даты для корректной обработки столбца ``timestamp``
        :param features_columns: имя столбца либо список имен столбцов с признаками
         для таблиц признаков пользователей/объектов.
         если не задан, в качестве признаков используются все столбцы датафрейма.
        :param kwargs: дополнительные аргументы, которые передаются в функцию
            ``spark.read.csv(path, **kwargs)``
        :return: спарк-датафрейм со столбцами, определенными в ``columns_names`` и features_columns
        """
        if data is not None:
            dataframe = convert2spark(data)
        elif path and format_type:
            dataframe = self._read_data(path, format_type, **kwargs)
        else:
            raise ValueError(
                "Один из параметров data, path должен быть отличным от None"
            )

        optional_columns = dict()

        if "user_id" in columns_names and "item_id" in columns_names:
            (
                features_columns,
                optional_columns,
                required_columns,
            ) = self.base_columns(features_columns)
        else:
            if len(columns_names) > 1:
                raise ValueError(
                    "Для датафрейма с признаками пользователей / объектов укажите в columns_names только"
                    " соответствие для текущего ключа (user_id или item_id)"
                )
            (features_columns, required_columns,) = self.feature_columns(
                columns_names, dataframe, features_columns  # type: ignore
            )

        self._check_columns(
            set(columns_names.keys()),
            required_columns=set(required_columns),
            optional_columns=set(optional_columns),
        )

        self._check_dataframe(dataframe, columns_names, features_columns)

        dataframe2 = self._rename_columns(
            dataframe,  # type: ignore
            columns_names,
            features_columns,
            default_schema={**required_columns, **optional_columns},
            date_format=date_format,
        )
        return dataframe2

    @staticmethod
    def feature_columns(
        columns_names: Dict[str, str],
        dataframe: DataFrame,
        features_columns: Union[str, Iterable[str], None],
    ) -> Tuple[List[str], Dict]:
        """Возвращает колонки для таблицы с фичами"""
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
            features_columns = sorted(
                list(dataframe_columns.difference(given_columns))
            )
            if not features_columns:
                raise ValueError("В датафрейме нет колонок с фичами")

        else:
            if isinstance(features_columns, str):
                features_columns = [features_columns]
            else:
                features_columns = list(features_columns)
        return features_columns, required_columns

    @staticmethod
    def base_columns(
        features_columns: Union[str, Iterable[str], None]
    ) -> Tuple[List, Dict, Dict]:
        """Возвращает колонки для лога"""
        required_columns = {
            "user_id": (None, StringType()),
            "item_id": (None, StringType()),
        }
        optional_columns = {
            "timestamp": ("1999-05-01", TimestampType()),
            "relevance": (1.0, DoubleType()),
        }
        if features_columns is None:
            features_columns = []
        else:
            raise ValueError("В данной таблице features не используются")
        return features_columns, optional_columns, required_columns


class CatFeaturesTransformer:
    """Преобразование категориальных признаков с использованием one-hot encoding.
    Может использоваться в двух режимах:
        1) Преобразование заданных колонок. Если передан список колонок для преобразования (cat_cols_list),
        к ним будет применен one-hot encoding, исходные колонки будут удалены.
        2) Поиск не числовых колонок и преобразование или удаление данных колонок по условию.
        После преобразования все колонки в датафрейме, кроме id пользователей и объектов, будут числовыми.
        Если cat_cols_list = None, будет выполнен поиск категориальных колонок среди всех,
        за исключением user_id/x, item_id/x.
        В случае, если задан threshold, колонки с большим, чем threshold числом уникальных значений будут удалены."""

    def __init__(
        self,
        cat_cols_list: Optional[List] = None,
        threshold: Optional[int] = None,
        alias: str = "ohe",
    ):
        """
        :param cat_cols_list: список категориальных колонок для преобразования
        :param threshold: количество уникальных значений колонки,
            при превышении которого колонка будет удалена из результирующего dataframe без преобразования.
            Не используется, если cat_cols_list явно передан.
        :param alias: префикс перед именем колонок, к которым применен one-hot encoding
        """
        self.cat_cols_list = cat_cols_list if cat_cols_list is not None else []
        self.cols_to_del = []
        self.expressions_list = []
        self.threshold = threshold
        self.alias = alias
        self.no_cols_left = False
        if cat_cols_list and threshold:
            State().logger.info(
                "threshold не будет применен, потому что передан cat_cols_list"
            )

    def fit(self, spark_df: Optional[DataFrame]) -> None:
        """
        Определение списка категориальных колонок, если не задан в init,
        сохранение списка категорий для каждой из колонок.
        :param spark_df: spark-датафрейм, содержащий категориальные и числовые признаки пользователей / объектов
        """
        if spark_df is None:
            self.no_cols_left = True
            return
        if not self.cat_cols_list:
            for col in spark_df.columns:
                if col not in ["user_idx", "item_idx", "user_id", "item_id"]:
                    if not isinstance(
                        spark_df.schema[col].dataType, NumericType
                    ):
                        if (
                            self.threshold is None
                            or spark_df.select(
                                sf.countDistinct(sf.col(col))
                            ).collect()[0][0]
                            <= self.threshold
                        ):
                            self.cat_cols_list.append(col)
                        else:
                            State().logger.warning(
                                "Колонка %s содержит более threshold уникальных "
                                "категориальных значений и будет удалена",
                                col,
                            )
                            self.cols_to_del.append(col)

        cat_feat_values_dict = {
            name: spark_df.select(name)
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect()
            for name in self.cat_cols_list
        }
        self.expressions_list = [
            sf.when(sf.col(col_name) == cur_name, 1)
            .otherwise(0)
            .alias(
                self.alias
                + "_"
                + str(col_name)[:10]
                + "_"
                + str(cur_name).translate(
                    str.maketrans(
                        "", "", string.punctuation + string.whitespace
                    )
                )[:30]
            )
            for col_name, col_values in cat_feat_values_dict.items()
            for cur_name in col_values
        ]
        if len(spark_df.columns) <= len(self.cols_to_del):
            self.no_cols_left = True

    def transform(self, spark_df: Optional[DataFrame]):
        """
        Преобразование категориальных колонок датафрейма.
        Новые значения категориальных переменных будут проигнорированы при преобразовании.
        :param spark_df: spark-датафрейм, содержащий категориальные и числовые признаки пользователей / объектов
        :return: spark-датафрейм после преобразования категориальных признаков
        """
        if spark_df is None or self.no_cols_left:
            return None
        return spark_df.select(*spark_df.columns, *self.expressions_list).drop(
            *(self.cols_to_del + self.cat_cols_list)
        )
