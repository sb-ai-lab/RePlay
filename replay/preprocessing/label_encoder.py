"""
Contains classes for encoding categorical data

``LabelEncodingRule`` to encode categorical data with value between 0 and n_classes-1 for single column.
    Recommended to use together with the LabelEncoder.
``LabelEncoder`` to apply multiple LabelEncodingRule to dataframe.
"""
import abc
import polars as pl
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Union

from replay.utils import (
    PYSPARK_AVAILABLE,
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
    get_spark_session,
)

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf
    from pyspark.storagelevel import StorageLevel
    from pyspark.sql.types import StructType, LongType

HandleUnknownStrategies = Literal["error", "use_default_value"]


# pylint: disable=missing-function-docstring
class BaseLabelEncodingRule(abc.ABC):  # pragma: no cover
    """
    Interface of the label encoding rule
    """

    @property
    @abc.abstractmethod
    def column(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_mapping(self) -> Mapping:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_inverse_mapping(self) -> Mapping:
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(self, df: DataFrameLike) -> "BaseLabelEncodingRule":
        raise NotImplementedError()

    @abc.abstractmethod
    def partial_fit(self, df: DataFrameLike) -> "BaseLabelEncodingRule":
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, df: DataFrameLike) -> DataFrameLike:
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, df: DataFrameLike) -> DataFrameLike:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_default_value(self, default_value: Optional[Union[int, str]]) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_handle_unknown(self, handle_unknown: HandleUnknownStrategies) -> None:
        raise NotImplementedError()


# pylint: disable=too-many-instance-attributes
class LabelEncodingRule(BaseLabelEncodingRule):
    """
    Implementation of the encoding rule for categorical variables of PySpark and Pandas Data Frames.
    Encodes target labels with value between 0 and n_classes-1 for the given column.
    It is recommended to use together with the LabelEncoder.
    """

    _ENCODED_COLUMN_SUFFIX: str = "_encoded"
    _HANDLE_UNKNOWN_STRATEGIES = ("error", "use_default_value")
    _TRANSFORM_PERFORMANCE_THRESHOLD_FOR_PANDAS = 100_000

    def __init__(
        self,
        column: str,
        mapping: Optional[Mapping] = None,
        handle_unknown: HandleUnknownStrategies = "error",
        default_value: Optional[Union[int, str]] = None,
    ):
        """
        :param column: Name of the column to encode.
        :param mapping: Prepared mapping for the column.
            If ``None``, then fit is necessary.
            If not ``None``, then the fit call does not change the already prepared mapping.
            Default: ``None``.
        :param handle_unknown:
            When set to ``error`` an error will be raised in case an unknown label is present during transform.
            When set to ``use_default_value``, the encoded value of unknown label will be set
            to the value given for the parameter default_value.
            Default: ``error``.
        :param default_value: Default value that will fill the unknown labels after transform.
            When the parameter handle_unknown is set to ``use_default_value``,
            this parameter is required and will set the encoded value of unknown labels.
            It has to be distinct from the values used to encode any of the labels in fit.
            If ``None``, then keep null.
            If ``int`` value, then fill by that value.
            If ``str`` value, should be \"last\" only, then fill by ``n_classes`` value.
            Default: ``None``.
        """
        if handle_unknown not in self._HANDLE_UNKNOWN_STRATEGIES:
            raise ValueError(f"handle_unknown should be either 'error' or 'use_default_value', got {handle_unknown}.")
        self._handle_unknown = handle_unknown
        if (
            self._handle_unknown == "use_default_value"
            and default_value is not None
            and not isinstance(default_value, int)
            and default_value != "last"
        ):
            raise ValueError("Default value should be None, int or 'last'")

        self._default_value = default_value
        self._col = column
        self._target_col = column + self._ENCODED_COLUMN_SUFFIX
        self._mapping = mapping
        self._is_fitted = False
        if self._mapping is not None:
            self._inverse_mapping = self._make_inverse_mapping()
            self._inverse_mapping_list = self._make_inverse_mapping_list()

    @property
    def column(self) -> str:
        return self._col

    def get_mapping(self) -> Mapping:
        if self._mapping is None:
            raise RuntimeError("Label encoder is not fitted")
        return self._mapping

    def get_inverse_mapping(self) -> Mapping:
        if self._mapping is None:
            raise RuntimeError("Label encoder is not fitted")
        return self._inverse_mapping

    def _make_inverse_mapping(self) -> Mapping:
        return {val: key for key, val in self.get_mapping().items()}

    def _make_inverse_mapping_list(self) -> List:
        inverse_mapping_list = [0 for _ in range(len(self.get_mapping()))]
        for k, value in self.get_mapping().items():
            inverse_mapping_list[value] = k
        return inverse_mapping_list

    def _fit_spark(self, df: SparkDataFrame) -> None:
        unique_col_values = df.select(self._col).distinct().persist(StorageLevel.MEMORY_ONLY)

        mapping_on_spark = (
            unique_col_values.rdd.zipWithIndex()
            .toDF(
                StructType()
                .add("_1",
                     StructType()
                     .add(self._col, df.schema[self._col].dataType, True),
                     True)
                .add("_2", LongType(), True)
            )
            .select(sf.col(f"_1.{self._col}").alias(self._col), sf.col("_2").alias(self._target_col))
            .persist(StorageLevel.MEMORY_ONLY)
        )
        mapping_on_spark.show()
        self._mapping = mapping_on_spark.rdd.collectAsMap()  # type: ignore
        mapping_on_spark.unpersist()
        unique_col_values.unpersist()

    def _fit_pandas(self, df: PandasDataFrame) -> None:
        unique_col_values = df[self._col].drop_duplicates().reset_index(drop=True)
        self._mapping = {val: key for key, val in unique_col_values.to_dict().items()}

    def _fit_polars(self, df: PolarsDataFrame) -> None:
        unique_col_values = df.select(self._col).unique()
        self._mapping = {key: val for val, key in enumerate(unique_col_values.to_series().to_list())}

    def fit(self, df: DataFrameLike) -> "LabelEncodingRule":
        """
        Fits encoder to input dataframe.

        :param df: input dataframe.
        :returns: fitted EncodingRule.
        """
        if self._mapping is not None:
            return self

        if isinstance(df, PandasDataFrame):
            self._fit_pandas(df)
        elif isinstance(df, SparkDataFrame):
            self._fit_spark(df)
        elif isinstance(df, PolarsDataFrame):
            self._fit_polars(df)
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(df)}")
        self._inverse_mapping = self._make_inverse_mapping()
        self._inverse_mapping_list = self._make_inverse_mapping_list()
        if self._handle_unknown == "use_default_value":
            if self._default_value in self._inverse_mapping:
                raise ValueError(
                    "The used value for default_value "
                    f"{self._default_value} is one of the "
                    "values already used for encoding the "
                    "seen labels."
                )
        self._is_fitted = True
        return self

    def _partial_fit_spark(self, df: SparkDataFrame) -> None:
        assert self._mapping is not None

        max_value = sf.lit(max(self._mapping.values()) + 1)
        already_fitted = list(self._mapping.keys())
        new_values = {x[self._col] for x in df.select(self._col).distinct().collect()} - set(already_fitted)
        new_values_list = [[x] for x in new_values]
        new_values_df: SparkDataFrame = get_spark_session().createDataFrame(new_values_list, schema=[self._col])
        new_unique_values = new_values_df.join(df, on=self._col, how="left").select(self._col)

        new_data: dict = (
            new_unique_values.rdd.zipWithIndex()
            .toDF(
                StructType()
                .add("_1",
                     StructType()
                     .add(self._col, df.schema[self._col].dataType),
                     True)
                .add("_2", LongType(), True)
            )
            .select(sf.col(f"_1.{self._col}").alias(self._col), sf.col("_2").alias(self._target_col))
            .withColumn(self._target_col, sf.col(self._target_col) + max_value)
            .rdd.collectAsMap()  # type: ignore
        )
        self._mapping.update(new_data)  # type: ignore
        self._inverse_mapping.update({v: k for k, v in new_data.items()})  # type: ignore
        self._inverse_mapping_list.extend(new_data.keys())
        new_unique_values.unpersist()

    def _partial_fit_pandas(self, df: PandasDataFrame) -> None:
        assert self._mapping is not None

        new_unique_values = set(df[self._col].tolist()) - set(self._mapping)
        new_data: dict = {value: max(self._mapping.values()) + i for i, value in enumerate(new_unique_values, start=1)}
        self._mapping.update(new_data)  # type: ignore
        self._inverse_mapping.update({v: k for k, v in new_data.items()})  # type: ignore
        self._inverse_mapping_list.extend(new_data.keys())

    def _partial_fit_polars(self, df: PolarsDataFrame) -> None:
        assert self._mapping is not None

        new_unique_values = set(df.select(self._col).unique().to_series().to_list()) - set(self._mapping)
        new_data: dict = {value: max(self._mapping.values()) + i for i, value in enumerate(new_unique_values, start=1)}
        self._mapping.update(new_data)  # type: ignore
        self._inverse_mapping.update({v: k for k, v in new_data.items()})  # type: ignore
        self._inverse_mapping_list.extend(new_data.keys())

    def partial_fit(self, df: DataFrameLike) -> "LabelEncodingRule":
        """
        Fits new data to already fitted encoder.

        :param df: input dataframe.
        :returns: fitted EncodingRule.
        """
        if self._mapping is None:
            return self.fit(df)

        if isinstance(df, SparkDataFrame):
            self._partial_fit_spark(df)
        elif isinstance(df, PandasDataFrame):
            self._partial_fit_pandas(df)
        elif isinstance(df, PolarsDataFrame):
            self._partial_fit_polars(df)
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(df)}")

        self._is_fitted = True
        return self

    def _transform_pandas(self, df: PandasDataFrame, default_value: Optional[int]) -> PandasDataFrame:
        is_unknown_label = False
        if df.shape[0] < self._TRANSFORM_PERFORMANCE_THRESHOLD_FOR_PANDAS:  # in order to speed up
            mapping = self.get_mapping()
            joined_df = df.copy()
            mapped_column = []
            for i in df[self._col]:
                val = mapping.get(i, -1)
                is_unknown_label |= val == -1
                mapped_column.append(val)
            joined_df[self._target_col] = mapped_column
        else:
            mapping_df = PandasDataFrame(self.get_mapping().items(), columns=[self._col, self._target_col])
            joined_df = df.merge(mapping_df, how="left", on=self._col)
            unknown_mask = joined_df[self._target_col].isna()
            joined_df.loc[unknown_mask, self._target_col] = -1
            is_unknown_label |= unknown_mask.sum() > 0

        if is_unknown_label and default_value != -1:
            unknown_mask = joined_df[self._target_col] == -1
            if self._handle_unknown == "error":
                unknown_unique_labels = joined_df[self._col][unknown_mask].unique().tolist()
                msg = f"Found unknown labels {unknown_unique_labels} in column {self._col} during transform"
                raise ValueError(msg)
            joined_df[self._target_col] = joined_df[self._target_col].astype("int")
            joined_df[self._target_col] = joined_df[self._target_col].replace({-1: default_value})

        result_df = joined_df.drop(self._col, axis=1).rename(columns={self._target_col: self._col})
        return result_df

    def _transform_spark(self, df: SparkDataFrame, default_value: Optional[int]) -> SparkDataFrame:
        mapping_on_spark = get_spark_session().createDataFrame(
            data=list(self.get_mapping().items()), schema=[self._col, self._target_col]
        )
        transformed_df = df.join(mapping_on_spark, on=self._col, how="left").withColumn(
            "unknown_mask", sf.isnull(self._target_col)
        )
        unknown_label_count = transformed_df.select(sf.sum(sf.col("unknown_mask").cast("long"))).first()[
            0
        ]  # type: ignore
        if unknown_label_count > 0:
            if self._handle_unknown == "error":
                collected_list = transformed_df.filter("unknown_mask == True").select(self._col).distinct().collect()
                unique_labels = [row[self._col] for row in collected_list]
                msg = f"Found unknown labels {unique_labels} in column {self._col} during transform"
                raise ValueError(msg)
            if default_value:
                transformed_df = transformed_df.fillna({self._target_col: default_value})

        result_df = transformed_df.drop(self._col, "unknown_mask").withColumnRenamed(self._target_col, self._col)
        return result_df

    def _transform_polars(self, df: PolarsDataFrame, default_value: Optional[int]) -> SparkDataFrame:
        mapping_on_polars = pl.from_records(
            [list(self.get_mapping().keys()), list(self.get_mapping().values())],
            schema=[self._col, self._target_col],
        )
        mapping_on_polars = mapping_on_polars.with_columns(
            pl.col(self._col).cast(df.get_column(self._col).dtype)
        )
        transformed_df = df.join(mapping_on_polars, on=self._col, how="left").with_columns(
            pl.col(self._target_col).is_null().alias("unknown_mask")
        )
        unknown_df = transformed_df.filter(pl.col("unknown_mask"))
        if not unknown_df.is_empty():
            if self._handle_unknown == "error":
                unique_labels = unknown_df.select(self._col).unique().to_series().to_list()
                msg = f"Found unknown labels {unique_labels} in column {self._col} during transform"
                raise ValueError(msg)
            if default_value:
                transformed_df = transformed_df.with_columns(pl.col(self._target_col).fill_null(default_value))

        result_df = transformed_df.drop([self._col, "unknown_mask"]).rename({self._target_col: self._col})
        return result_df

    def transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Transforms input dataframe with fitted encoder.

        :param df: input dataframe.
        :returns: transformed dataframe.
        """
        if self._mapping is None:
            raise RuntimeError("Label encoder is not fitted")

        default_value = len(self._mapping) if self._default_value == "last" else self._default_value

        if isinstance(df, PandasDataFrame):
            transformed_df = self._transform_pandas(df, default_value)  # type: ignore
        elif isinstance(df, SparkDataFrame):
            transformed_df = self._transform_spark(df, default_value)  # type: ignore
        elif isinstance(df, PolarsDataFrame):
            transformed_df = self._transform_polars(df, default_value)  # type: ignore
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(df)}")
        return transformed_df

    def _inverse_transform_pandas(self, df: PandasDataFrame) -> PandasDataFrame:
        dff = df.copy()
        dff[self._col] = [self._inverse_mapping_list[i] for i in df[self._col]]
        return dff

    def _inverse_transform_spark(self, df: SparkDataFrame) -> SparkDataFrame:
        mapping_on_spark = get_spark_session().createDataFrame(
            data=list(self.get_mapping().items()), schema=[self._col, self._target_col]
        )
        transformed_df = (
            df.withColumnRenamed(self._col, self._target_col)
            .join(mapping_on_spark, on=self._target_col, how="left")
            .drop(self._target_col)
        )
        return transformed_df

    def _inverse_transform_polars(self, df: PolarsDataFrame) -> PolarsDataFrame:
        mapping_on_polars = pl.from_records(
            [list(self.get_mapping().keys()), list(self.get_mapping().values())],
            schema=[self._col, self._target_col],
        )
        transformed_df = (
            df.rename({self._col: self._target_col})
            .join(mapping_on_polars, on=self._target_col, how="left")
            .drop(self._target_col)
        )
        return transformed_df

    def inverse_transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Reverse transform of transformed dataframe.

        :param df: transformed dataframe.
        :returns: initial dataframe.
        """
        if self._mapping is None:
            raise RuntimeError("Label encoder is not fitted")

        if isinstance(df, PandasDataFrame):
            transformed_df = self._inverse_transform_pandas(df)
        elif isinstance(df, SparkDataFrame):
            transformed_df = self._inverse_transform_spark(df)
        elif isinstance(df, PolarsDataFrame):
            transformed_df = self._inverse_transform_polars(df)
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not implemented for {type(df)}")
        return transformed_df

    def set_default_value(self, default_value: Optional[Union[int, str]]) -> None:
        """
        Sets default value to deal with unknown labels.
        Used when handle_unknown_strategy is 'use_default_value'.

        :param default_value: default value.
        """
        if default_value is not None and not isinstance(default_value, int) and default_value != "last":
            raise ValueError("Default value should be None, int or 'last'")
        self._default_value = default_value

    def set_handle_unknown(self, handle_unknown: HandleUnknownStrategies) -> None:
        """
        Sets strategy to handle unknown labels.

        :param handle_unknown: handle unknown strategy.
        """
        if handle_unknown not in self._HANDLE_UNKNOWN_STRATEGIES:
            raise ValueError(f"handle_unknown should be either 'error' or 'use_default_value', got {handle_unknown}.")
        self._handle_unknown = handle_unknown


class LabelEncoder:
    """
    Applies multiple label encoding rules to the data frame.

    >>> import pandas as pd
    >>> user_interactions = pd.DataFrame([
    ...    ("u1", "item_1", "item_1"),
    ...    ("u2", "item_2", "item_2"),
    ...    ("u3", "item_3", "item_3"),
    ... ], columns=["user_id", "item_1", "item_2"])
    >>> user_interactions
      user_id  item_1  item_2
    0      u1  item_1  item_1
    1      u2  item_2  item_2
    2      u3  item_3  item_3
    >>> encoder = LabelEncoder(
    ...    [LabelEncodingRule("user_id"), LabelEncodingRule("item_1"), LabelEncodingRule("item_2")]
    ... )
    >>> mapped_interactions = encoder.fit_transform(user_interactions)
    >>> mapped_interactions
       user_id  item_1  item_2
    0        0       0       0
    1        1       1       1
    2        2       2       2
    >>> encoder.mapping
    {'user_id': {'u1': 0, 'u2': 1, 'u3': 2},
    'item_1': {'item_1': 0, 'item_2': 1, 'item_3': 2},
    'item_2': {'item_1': 0, 'item_2': 1, 'item_3': 2}}
    >>> encoder.inverse_mapping
    {'user_id': {0: 'u1', 1: 'u2', 2: 'u3'},
    'item_1': {0: 'item_1', 1: 'item_2', 2: 'item_3'},
    'item_2': {0: 'item_1', 1: 'item_2', 2: 'item_3'}}
    >>> new_encoder = LabelEncoder([
    ...    LabelEncodingRule("user_id", encoder.mapping["user_id"]),
    ...    LabelEncodingRule("item_1", encoder.mapping["item_1"]),
    ...    LabelEncodingRule("item_2", encoder.mapping["item_2"])
    ... ])
    >>> new_encoder.inverse_transform(mapped_interactions)
      user_id  item_1  item_2
    0      u1  item_1  item_1
    1      u2  item_2  item_2
    2      u3  item_3  item_3
    <BLANKLINE>
    """

    def __init__(self, rules: Sequence[BaseLabelEncodingRule]):
        """
        :param rules: Sequence of rules.
        """
        self.rules = rules

    @property
    def mapping(self) -> Mapping[str, Mapping]:
        """
        Returns mapping of each column in given rules.
        """
        return {r.column: r.get_mapping() for r in self.rules}

    @property
    def inverse_mapping(self) -> Mapping[str, Mapping]:
        """
        Returns inverse mapping of each column in given rules.
        """
        return {r.column: r.get_inverse_mapping() for r in self.rules}

    def fit(self, df: DataFrameLike) -> "LabelEncoder":
        """
        Fits an encoder by the input data frame with given rules.

        :param df: input dataframe.
        :returns: fitted LabelEncoder.
        """
        for rule in self.rules:
            rule.fit(df)
        return self

    def partial_fit(self, df: DataFrameLike) -> "LabelEncoder":
        """
        Fits an already fitted encoder by the new input data frame with given rules.
        If encoder has not been fitted yet - performs default fit.

        :param df: input dataframe.
        :returns: fitted LabelEncoder.
        """
        for rule in self.rules:
            rule.partial_fit(df)
        return self

    def transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Transforms the input data frame.
        If the input data frame contains unknown labels then they will be transformed by handle unknown strategy.

        :param df: input dataframe.
        :returns: transformed dataframe.
        """
        for rule in self.rules:
            df = rule.transform(df)
        return df

    def inverse_transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Performs inverse transform of the input data frame.

        :param df: input dataframe.
        :returns: initial dataframe.
        """
        for rule in self.rules:
            df = rule.inverse_transform(df)
        return df

    def fit_transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Fits an encoder by the input data frame with given rules and transforms the input data frame.

        :param df: input dataframe.
        :returns: transformed dataframe.
        """
        return self.fit(df).transform(df)

    def set_handle_unknowns(self, handle_unknown_rules: Dict[str, HandleUnknownStrategies]) -> None:
        """
        Modify handle unknown strategy on already fitted encoder.

        :param handle_unknown_rules: handle unknown rule.

        Example: {"item_id" : None, "user_id" : -1, "category_column" : "last"}

        Default value examples:
            If ``None``, then keep null.
            If ``int`` value, then fill by that value.
            If ``str`` value, should be \"last\" only, then fill by n_classes number.
            Default ``None``.
        """
        columns = [i.column for i in self.rules]  # pylint: disable=W0212
        for column, handle_unknown in handle_unknown_rules.items():
            if column not in columns:
                raise ValueError(f"Column {column} not found.")
            rule = list(filter(lambda x: x.column == column, self.rules))  # pylint: disable = W0212, W0640
            rule[0].set_handle_unknown(handle_unknown)

    def set_default_values(self, default_value_rules: Dict[str, Optional[Union[int, str]]]) -> None:
        """
        Modify handle unknown strategy on already fitted encoder.
        Default value that will fill the unknown labels
        after transform if handle_unknown is set to ``use_default_value``.

        :param default_value_rules: Dictionary of default values to set for columns.

        Example: {"item_id" : "error", "user_id" : "use_default_value"}

        Default value examples:
            When set to ``error`` an error will be raised in case an unknown label is present during transform.
            When set to ``use_default_value``, the encoded value of unknown label will be set
            to the value given for the parameter default_value.
            Default: ``error``.
        """
        columns = [i.column for i in self.rules]  # pylint: disable=W0212
        for column, default_value in default_value_rules.items():
            if column not in columns:
                raise ValueError(f"Column {column} not found.")
            rule = list(filter(lambda x: x.column == column, self.rules))  # pylint: disable = W0212, W0640
            rule[0].set_default_value(default_value)
