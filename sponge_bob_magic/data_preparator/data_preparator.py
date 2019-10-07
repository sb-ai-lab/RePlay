import collections
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import FloatType, StringType, TimestampType
from sponge_bob_magic import constants


def flat_list(l: Iterable):
    for el in l:
        if (
                isinstance(el, collections.abc.Iterable) and
                not isinstance(el, (str, bytes))
        ):
            yield from flat_list(el)
        else:
            yield el


class DataPreparator:
    spark: SparkSession

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def _read_data(self,
                   path: str,
                   format_type: str,
                   **kwargs) -> DataFrame:
        if format_type == 'csv':
            df = self.spark.read.csv(path, **kwargs)
        elif format_type == 'parquet':
            df = self.spark.read.parquet(path)
        elif format_type == 'json':
            df = self.spark.read.json(path, **kwargs)
        elif format_type == 'table':
            df = self.spark.read.table(path)
        else:
            raise ValueError(f"Invalid value of format_type='{format_type}'")

        return df

    @staticmethod
    def _check_columns(given_columns: Set[str],
                       required_columns: Set[str],
                       optional_columns: Set[str]):
        if not required_columns.issubset(given_columns):
            raise ValueError(
                f"В датафрейме нет обязательных колонок ({required_columns})")
        if len(
                given_columns
                        .difference(required_columns)
                        .difference(optional_columns)
        ) > 0:
            raise ValueError("В 'columns_names' есть лишние колонки")

    @staticmethod
    def _check_dataframe(df: DataFrame,
                         columns_names: Dict[str, Union[str, List[str]]]):
        # чекаем, что датафрейм не пустой
        if len(df.head(1)) == 0:
            raise ValueError("Датафрейм пустой")

        # чекаем, что данные юзером колонки реально есть в датафрейме
        given_columns = set((flat_list(list(columns_names.values()))))
        dataframe_columns = set(df.columns)
        if not given_columns.issubset(dataframe_columns):
            raise ValueError(
                "В columns_names есть колонки, которых нет в датафрейме; "
                f"columns_names={given_columns}, "
                f"df.columns={dataframe_columns}")

        # чекаем на нуллы
        for column in given_columns:
            if df.where(sf.col(column).isNull()).count() > 0:
                raise ValueError(f"В колонке '{column}' есть значения NULL")

    @staticmethod
    def _rename_columns(df: DataFrame,
                        columns_names: Dict[str, str or List[str]],
                        default_schema: Dict[str, Tuple],
                        date_format: Optional[str] = None):
        # колонки с фичами не будут переименованы, их надо просто селектить
        features_columns = []
        if 'features' in columns_names:
            features_columns = columns_names['features']
            if type(features_columns) is not list:
                features_columns = [features_columns]

        # переименовываем колонки
        df = df.select([sf.col(column).alias(new_name)
                        for new_name, column in
                        columns_names.items()
                        if new_name != 'features'] +
                       features_columns)

        # добавляем необязательные дефолтные колонки, если их нет,
        # и задаем тип для тех колонок, что есть
        for column_name, (default_value,
                          default_type) in default_schema.items():
            if column_name not in df.columns:
                column = sf.lit(default_value)
            else:
                column = sf.col(column_name)
            if column_name == 'timestamp':
                df = df.withColumn(column_name,
                                   sf.to_timestamp(column, format=date_format))
            else:
                df = df.withColumn(column_name, column.cast(default_type))
        return df

    def transform_log(self,
                      path: str,
                      format_type: str,
                      columns_names: dict,
                      date_format: Optional[str] = None,
                      **kwargs) -> DataFrame:
        """
        Преобразовывает лог формата `format_type` в файле по пути `path` в
        спарк-датафрейм вида (user_id, item_id, timestamp, context, relevance).
        :param path: путь к файлу с логом
        :param format_type: тип файла, принимает значения из списка
            [`csv`, `parquet`, `json`, `table`]
        :param columns_names: маппинг колонок, ключ - значения из списка
            [`user_id`, `item_id`, `timestamp`, `context`, `relevance`];
            обязательными являются только [`user_id`, `item_id`];
            значения - колонки в логе
        :param date_format: формат даты, нужен, если формат даты особенный
        :param kwargs: дополнительные аргументы, которые передаются в функцию
            `spark.read.csv(path, **kwargs)`
        :return: спарк-датафрейм с колонками
            [`user_id`, `item_id`, `timestamp`, `context`, `relevance`];
            колонки, не предоставленные в `columns_names`,
            заполянются дефолтными значениями
        """
        self._check_columns(set(columns_names.keys()),
                            required_columns={'user_id', 'item_id'},
                            optional_columns={'timestamp', 'context',
                                              'relevance'})

        df = self._read_data(path, format_type, **kwargs)
        self._check_dataframe(df, columns_names)

        log_schema = {
            'timestamp': ('1999-05-01', TimestampType()),
            'context': (constants.DEFAULT_CONTEXT, StringType()),
            'relevance': (1.0, FloatType()),
            'user_id': (None, StringType()),
            'item_id': (None, StringType()),
        }
        df = self._rename_columns(df, columns_names,
                                  default_schema=log_schema,
                                  date_format=date_format)

        return df

    def transform_features(self,
                           path: str,
                           format_type: str,
                           columns_names: dict,
                           date_format: Optional[str] = None,
                           **kwargs) -> DataFrame:
        """
        Преобразовывает признаки формата `format_type` в файле по пути `path` в
        спарк-датафрейм вида
        (user_id, timestamp, features) или (item_id, timestamp, features).
        :param path: путь к файлу с признаками
        :param format_type: тип файла, принимает значения из списка
            [`csv`, `parquet`, `json`, `table`]
        :param columns_names: маппинг колонок, ключ - значения из списка
            [`user_id` / `item_id`, `timestamp`, `features`];
            обязательными являются только
            [`user_id`] или [`item_id`] (должен быть один из них);
            если `features` нет в ключах,
            то призанками фичей явлются все оставшиеся колонки;
            в качестве `features` может подаваться как список, так и отдельное
            значение колонки (если признак один);
            значения - колонки в табличке признаков
        :param date_format: формат даты, нужен,
            если формат колонки `timestamp` особенный
        :param kwargs: дополнительные аргументы, которые передаются в функцию
            `spark.read.csv(path, **kwargs)`
        :return: спарк-датафрейм с колонками
            [`user_id` / `item_id`, `timestamp`] и колонки с признаками;
            колонка `timestamp`, если ее нет в `columns_names`,
            заполянются дефолтными значениями
        """
        if 'user_id' in columns_names:
            required_columns = {'user_id'}
        elif 'item_id' in columns_names:
            required_columns = {'item_id'}
        else:
            raise ValueError("В columns_names нет ни 'user_id', ни 'item_id'")

        self._check_columns(set(columns_names.keys()),
                            required_columns=required_columns,
                            optional_columns={'timestamp', 'features'})

        df = self._read_data(path, format_type, **kwargs)

        # если фичей нет в данных юзером колонках, вставляем все оставшиеся
        # нужно, чтобы проверить, что там нет нуллов
        if 'features' not in columns_names:
            given_columns = flat_list(list(columns_names.values()))
            df_columns = df.columns
            feature_columns = list(set(df_columns).difference(given_columns))
            if len(feature_columns) == 0:
                raise ValueError("В датафрейме нет колонок с фичами")
            features_dict = {'features': feature_columns}
        else:
            features_dict = dict()

        self._check_dataframe(df, {**columns_names, **features_dict})

        features_schema = {
            'timestamp': ('1999-05-01', TimestampType()),
            ('user_id' if 'user_id' in columns_names else 'item_id'):
                (None, StringType()),
        }

        df = self._rename_columns(df, columns_names,
                                  default_schema=features_schema,
                                  date_format=date_format)
        return df
