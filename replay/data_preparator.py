"""
Contains classes ``DataPreparator`` and ``CatFeaturesTransformer``.
``DataPreparator`` is used to transform DataFrames to a library format.

`CatFeaturesTransformer`` transforms cateforical features with one-hot encoding.
"""
import string
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    DoubleType,
    StringType,
    TimestampType,
)

from replay.constants import AnyDataFrame
from replay.utils import convert2spark, process_timestamp_column
from replay.session_handler import State


# pylint: disable=too-few-public-methods
class DataPreparator:
    """Transforms data to a library format. If both user id and item id are provided, DataFrame
    is trated as a log, otherwise as a feature DataFrame.

    Examples:

    Loading log DataFrame

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


    Loading user features

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

    If feature_columns is not set, all extra columns are trated as feature columns

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
                "Required columns are missing: "
                f"{required_columns.difference(given_columns)}"
            )

        excess_columns = given_columns.difference(required_columns).difference(
            optional_columns
        )
        if excess_columns:
            raise ValueError(
                "'columns_names' has excess columns: " f"{excess_columns}"
            )

    @staticmethod
    def _check_dataframe(
        dataframe: DataFrame,
        columns_names: Dict[str, str],
        feature_columns: List[str],
    ):
        if not dataframe.head(1):
            raise ValueError("DataFrame is empty")

        columns_to_check = {*columns_names.values(), *feature_columns}
        dataframe_columns = set(dataframe.columns)
        if not columns_to_check.issubset(dataframe_columns):
            raise ValueError(
                "feature_columns or columns_names has columns that are not present in DataFrame "
                f"{columns_to_check.difference(dataframe_columns)}"
            )

        for column in columns_names.values():
            if dataframe.where(sf.col(column).isNull()).count() > 0:
                raise ValueError(f"Column '{column}' has NULL values")

    @staticmethod
    def _rename_columns(
        dataframe: DataFrame,
        columns_names: Dict[str, str],
        features_columns: List[str],
        default_schema: Dict[str, Tuple[Any, Any]],
        date_format: Optional[str] = None,
    ):
        dataframe = dataframe.select(
            [
                sf.col(column).alias(new_name)
                for new_name, column in columns_names.items()
            ]
            + [sf.col(column) for column in features_columns]
        )

        for (
            column_name,
            (default_value, default_type),
        ) in default_schema.items():
            if column_name not in dataframe.columns:
                dataframe = dataframe.withColumn(
                    column_name, sf.lit(default_value)
                )
            if column_name == "timestamp":
                dataframe = process_timestamp_column(
                    dataframe, column_name, date_format
                )
            else:
                dataframe = dataframe.withColumn(
                    column_name, sf.col(column_name).cast(default_type)
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
        reader_kwargs: Optional[Dict] = None,
    ) -> DataFrame:
        """
        Transforms log, user or item features into a Spark DataFrame
        ``[user_id, user_id, timestamp, relevance]``,
        ``[user_id, *features]``, or  ``[item_id, *features]``.
        Input is either file of ``format_type``
        at ``path``, or ``pandas.DataFrame`` or ``spark.DataFrame``.
        :param columns_names: dictionary mapping "default column name:
        column name in input DataFrame"
        ``user_id`` and ``item_id`` mappings are required,
        ``timestamp`` and``relevance`` are optional.

            Mapping specifies the meaning of the DataFrame:
            - Both ``[user_id, item_id]`` are present, then it's log
            - Only ``[user_id]`` is present, then it's user features
            - Only ``[item_id]`` is present, then it's item features

        :param data: DataFrame to process
        :param path: path to data
        :param format_type: file type, one of ``[csv , parquet , json , table]``
        :param date_format: format for the ``timestamp``
        :param features_columns: names of the feature columns
        :param reader_kwargs: extra arguments passed to
            ``spark.read.<format>(path, **reader_kwargs)``
        :return: processed DataFrame
        """
        if data is not None:
            dataframe = convert2spark(data)
        elif path and format_type:
            if reader_kwargs is None:
                reader_kwargs = {}
            dataframe = self._read_data(path, format_type, **reader_kwargs)
        else:
            raise ValueError("Either data or path parameters must not be None")

        optional_columns = {}

        if "user_id" in columns_names and "item_id" in columns_names:
            (
                features_columns,
                optional_columns,
                required_columns,
            ) = self.base_columns(features_columns)
        else:
            if len(columns_names) > 1:
                raise ValueError(
                    "Feature DataFrame mappings must contain mapping only for one id, user or item."
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
        """Get feature columns"""
        if "user_id" in columns_names:
            required_columns = {"user_id": (None, StringType())}
        elif "item_id" in columns_names:
            required_columns = {"item_id": (None, StringType())}
        else:
            raise ValueError(
                "columns_names have neither 'user_id', nor 'item_id'"
            )

        if features_columns is None:
            given_columns = set(columns_names.values())
            dataframe_columns = set(dataframe.columns)
            features_columns = sorted(
                list(dataframe_columns.difference(given_columns))
            )
            if not features_columns:
                raise ValueError("Feature columns missing")

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
        """Get base columns"""
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
            raise ValueError("features are not used")
        return features_columns, optional_columns, required_columns


class CatFeaturesTransformer:
    """Transform categorical features in ``cat_cols_list``
    with one-hot encoding and delete other columns."""

    def __init__(
        self,
        cat_cols_list: List,
        threshold: Optional[int] = None,
        alias: str = "ohe",
    ):
        """
        :param cat_cols_list: list of categorical columns
        :param alias: prefix for one-hot encoding columns
        """
        self.cat_cols_list = cat_cols_list
        self.expressions_list = []
        self.threshold = threshold
        self.alias = alias
        if threshold is not None:
            State().logger.info(
                "threshold не будет применен, функциональность в разработке"
            )

    def fit(self, spark_df: Optional[DataFrame]) -> None:
        """
        Save categories for each column
        :param spark_df: Spark DataFrame with features
        """
        if spark_df is None:
            return

        cat_feat_values_dict = {
            name: (
                spark_df.select(sf.collect_set(sf.col(name))).collect()[0][0]
            )
            for name in self.cat_cols_list
        }
        self.expressions_list = [
            sf.when(sf.col(col_name) == cur_name, 1)
            .otherwise(0)
            .alias(
                f"""{self.alias}_{col_name}_{str(cur_name).translate(
                        str.maketrans(
                            "", "", string.punctuation + string.whitespace
                        )
                    )[:30]}"""
            )
            for col_name, col_values in cat_feat_values_dict.items()
            for cur_name in col_values
        ]

    def transform(self, spark_df: Optional[DataFrame]):
        """
        Transform categorical columns.
        If there are any new categories that were not present at fit stage, they will be ignored.
        :param spark_df: feature DataFrame
        :return: transformed DataFrame
        """
        if spark_df is None:
            return None
        return spark_df.select(*spark_df.columns, *self.expressions_list).drop(
            *self.cat_cols_list
        )
