"""
Contains classes for encoding categorical data

``LabelEncodingRule`` to encode categorical data with value between 0 and n_classes-1 for single column.
    Recommended to use together with the LabelEncoder.
``LabelEncoder`` to apply multiple LabelEncodingRule to dataframe.
"""

import abc
import json
import os
import warnings
from itertools import chain
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Union

import polars as pl

from replay.utils import (
    PYSPARK_AVAILABLE,
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
    get_spark_session,
)

if PYSPARK_AVAILABLE:
    from pyspark.sql import Window, functions as sf  # noqa: I001
    from pyspark.sql.types import LongType

HandleUnknownStrategies = Literal["error", "use_default_value", "drop"]


class LabelEncoderTransformWarning(Warning):
    """Label encoder transform warning."""


class LabelEncoderPartialFitWarning(Warning):
    """Label encoder partial fit warning."""


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


class LabelEncodingRule(BaseLabelEncodingRule):
    """
    Implementation of the encoding rule for categorical variables of PySpark, Pandas and Polars Data Frames.
    Encodes target labels with value between 0 and n_classes-1 for the given column.
    It is recommended to use together with the LabelEncoder.
    """

    _ENCODED_COLUMN_SUFFIX: str = "_encoded"
    _HANDLE_UNKNOWN_STRATEGIES = ("error", "use_default_value", "drop")
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
            When set to ``drop``, the unknown labels will be dropped.
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
            msg = f"handle_unknown should be either 'error' or 'use_default_value', got {handle_unknown}."
            raise ValueError(msg)
        self._handle_unknown = handle_unknown
        if (
            self._handle_unknown == "use_default_value"
            and default_value is not None
            and not isinstance(default_value, int)
            and default_value != "last"
        ):
            msg = "Default value should be None, int or 'last'"
            raise ValueError(msg)

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
            msg = "Label encoder is not fitted"
            raise RuntimeError(msg)
        return self._mapping

    def get_inverse_mapping(self) -> Mapping:
        if self._mapping is None:
            msg = "Label encoder is not fitted"
            raise RuntimeError(msg)
        return self._inverse_mapping

    def _make_inverse_mapping(self) -> Mapping:
        return {val: key for key, val in self.get_mapping().items()}

    def _make_inverse_mapping_list(self) -> List:
        inverse_mapping_list = [0 for _ in range(len(self.get_mapping()))]
        for k, value in self.get_mapping().items():
            inverse_mapping_list[value] = k
        return inverse_mapping_list

    def _fit_spark(self, df: SparkDataFrame) -> None:
        unique_col_values = df.select(self._col).distinct()
        window_function_give_ids = Window.orderBy(self._col)

        mapping_on_spark = (
            unique_col_values.withColumn(
                self._target_col,
                sf.row_number().over(window_function_give_ids).cast(LongType()),
            )
            .withColumn(self._target_col, sf.col(self._target_col) - 1)
            .select(self._col, self._target_col)
        )

        self._mapping = mapping_on_spark.rdd.collectAsMap()

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
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)
        self._inverse_mapping = self._make_inverse_mapping()
        self._inverse_mapping_list = self._make_inverse_mapping_list()
        if self._handle_unknown == "use_default_value" and self._default_value in self._inverse_mapping:
            msg = (
                "The used value for default_value "
                f"{self._default_value} is one of the "
                "values already used for encoding the "
                "seen labels."
            )
            raise ValueError(msg)
        self._is_fitted = True
        return self

    def _partial_fit_spark(self, df: SparkDataFrame) -> None:
        assert self._mapping is not None
        max_value = sf.lit(max(self._mapping.values()) + 1)
        already_fitted = list(self._mapping.keys())
        new_values = {x[self._col] for x in df.select(self._col).distinct().collect()} - set(already_fitted)
        new_values_list = [[x] for x in new_values]
        if len(new_values_list) == 0:
            warnings.warn(
                "partial_fit will have no effect because "
                f"there are no new values in the incoming dataset at '{self.column}' column",
                LabelEncoderPartialFitWarning,
            )
            return
        new_unique_values_df: SparkDataFrame = get_spark_session().createDataFrame(new_values_list, schema=[self._col])
        window_function_give_ids = Window.orderBy(self._col)
        new_part_of_mapping = (
            new_unique_values_df.withColumn(
                self._target_col,
                sf.row_number().over(window_function_give_ids).cast(LongType()),
            )
            .withColumn(self._target_col, sf.col(self._target_col) - 1 + max_value)
            .select(self._col, self._target_col)
            .rdd.collectAsMap()
        )
        self._mapping.update(new_part_of_mapping)
        self._inverse_mapping.update({v: k for k, v in new_part_of_mapping.items()})
        self._inverse_mapping_list.extend(new_part_of_mapping.keys())

    def _partial_fit_pandas(self, df: PandasDataFrame) -> None:
        assert self._mapping is not None

        new_unique_values = set(df[self._col].tolist()) - set(self._mapping)
        if len(new_unique_values) == 0:
            warnings.warn(
                "partial_fit will have no effect because "
                f"there are no new values in the incoming dataset at '{self.column}' column",
                LabelEncoderPartialFitWarning,
            )
            return
        last_mapping_value = max(self._mapping.values())
        new_data: dict = {value: last_mapping_value + i for i, value in enumerate(new_unique_values, start=1)}
        self._mapping.update(new_data)
        self._inverse_mapping.update({v: k for k, v in new_data.items()})
        self._inverse_mapping_list.extend(new_data.keys())

    def _partial_fit_polars(self, df: PolarsDataFrame) -> None:
        assert self._mapping is not None

        new_unique_values = set(df.select(self._col).unique().to_series().to_list()) - set(self._mapping)
        if len(new_unique_values) == 0:
            warnings.warn(
                "partial_fit will have no effect because "
                f"there are no new values in the incoming dataset at '{self.column}' column",
                LabelEncoderPartialFitWarning,
            )
            return
        new_data: dict = {value: max(self._mapping.values()) + i for i, value in enumerate(new_unique_values, start=1)}
        self._mapping.update(new_data)
        self._inverse_mapping.update({v: k for k, v in new_data.items()})
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
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)

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

        if is_unknown_label:
            unknown_mask = joined_df[self._target_col] == -1
            if self._handle_unknown == "drop":
                joined_df.drop(joined_df[unknown_mask].index, inplace=True)
                if joined_df.empty:
                    warnings.warn(
                        f"You are trying to transform dataframe with all values are unknown for {self._col}, "
                        "with `handle_unknown_strategy=drop` leads to empty dataframe",
                        LabelEncoderTransformWarning,
                    )
            elif self._handle_unknown == "error":
                unknown_unique_labels = joined_df[self._col][unknown_mask].unique().tolist()
                msg = f"Found unknown labels {unknown_unique_labels} in column {self._col} during transform"
                raise ValueError(msg)
            else:
                if default_value != -1:
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
        unknown_label_count = transformed_df.select(sf.sum(sf.col("unknown_mask").cast("long"))).first()[0]
        if unknown_label_count > 0:
            if self._handle_unknown == "drop":
                transformed_df = transformed_df.filter("unknown_mask == False")
                if transformed_df.rdd.isEmpty():
                    warnings.warn(
                        f"You are trying to transform dataframe with all values are unknown for {self._col}, "
                        "with `handle_unknown_strategy=drop` leads to empty dataframe",
                        LabelEncoderTransformWarning,
                    )
            elif self._handle_unknown == "error":
                collected_list = transformed_df.filter("unknown_mask == True").select(self._col).distinct().collect()
                unique_labels = [row[self._col] for row in collected_list]
                msg = f"Found unknown labels {unique_labels} in column {self._col} during transform"
                raise ValueError(msg)
            else:
                if default_value:
                    transformed_df = transformed_df.fillna({self._target_col: default_value})

        result_df = transformed_df.drop(self._col, "unknown_mask").withColumnRenamed(self._target_col, self._col)
        return result_df

    def _transform_polars(self, df: PolarsDataFrame, default_value: Optional[int]) -> SparkDataFrame:
        mapping_on_polars = pl.from_records(
            [list(self.get_mapping().keys()), list(self.get_mapping().values())],
            schema=[self._col, self._target_col],
        )
        mapping_on_polars = mapping_on_polars.with_columns(pl.col(self._col).cast(df.get_column(self._col).dtype))
        transformed_df = df.join(mapping_on_polars, on=self._col, how="left").with_columns(
            pl.col(self._target_col).is_null().alias("unknown_mask")
        )
        unknown_df = transformed_df.filter(pl.col("unknown_mask"))
        if not unknown_df.is_empty():
            if self._handle_unknown == "drop":
                transformed_df = transformed_df.filter(pl.col("unknown_mask") == "false")
                if transformed_df.is_empty():
                    warnings.warn(
                        f"You are trying to transform dataframe with all values are unknown for {self._col}, "
                        "with `handle_unknown_strategy=drop` leads to empty dataframe",
                        LabelEncoderTransformWarning,
                    )
            elif self._handle_unknown == "error":
                unique_labels = unknown_df.select(self._col).unique().to_series().to_list()
                msg = f"Found unknown labels {unique_labels} in column {self._col} during transform"
                raise ValueError(msg)
            else:
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
            msg = "Label encoder is not fitted"
            raise RuntimeError(msg)

        default_value = len(self._mapping) if self._default_value == "last" else self._default_value

        if isinstance(df, PandasDataFrame):
            transformed_df = self._transform_pandas(df, default_value)
        elif isinstance(df, SparkDataFrame):
            transformed_df = self._transform_spark(df, default_value)
        elif isinstance(df, PolarsDataFrame):
            transformed_df = self._transform_polars(df, default_value)
        else:
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)
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
            msg = "Label encoder is not fitted"
            raise RuntimeError(msg)

        if isinstance(df, PandasDataFrame):
            transformed_df = self._inverse_transform_pandas(df)
        elif isinstance(df, SparkDataFrame):
            transformed_df = self._inverse_transform_spark(df)
        elif isinstance(df, PolarsDataFrame):
            transformed_df = self._inverse_transform_polars(df)
        else:
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)
        return transformed_df

    def set_default_value(self, default_value: Optional[Union[int, str]]) -> None:
        """
        Sets default value to deal with unknown labels.
        Used when handle_unknown_strategy is 'use_default_value'.

        :param default_value: default value.
        """
        if default_value is not None and not isinstance(default_value, int) and default_value != "last":
            msg = "Default value should be None, int or 'last'"
            raise ValueError(msg)
        self._default_value = default_value

    def set_handle_unknown(self, handle_unknown: HandleUnknownStrategies) -> None:
        """
        Sets strategy to handle unknown labels.

        :param handle_unknown: handle unknown strategy.
        """
        if handle_unknown not in self._HANDLE_UNKNOWN_STRATEGIES:
            msg = f"handle_unknown should be either 'error' or 'use_default_value', got {handle_unknown}."
            raise ValueError(msg)
        self._handle_unknown = handle_unknown

    def save(
        self,
        path: str,
    ) -> None:
        encoder_rule_dict = {}
        encoder_rule_dict["_class_name"] = self.__class__.__name__
        encoder_rule_dict["init_args"] = {
            "column": self._col,
            "mapping": self._mapping,
            "handle_unknown": self._handle_unknown,
            "default_value": self._default_value,
        }

        column_type = str(type(next(iter(self._mapping))))

        if not isinstance(column_type, (str, int, float)):  # pragma: no cover
            msg = f"LabelEncodingRule.save() is not implemented for column type {column_type}. \
Convert type to string, integer, or float."
            raise NotImplementedError(msg)

        encoder_rule_dict["fitted_args"] = {
            "target_col": self._target_col,
            "is_fitted": self._is_fitted,
            "column_type": column_type,
        }

        base_path = Path(path).with_suffix(".replay").resolve()
        if os.path.exists(base_path):  # pragma: no cover
            msg = "There is already LabelEncodingRule object saved at the given path. File will be overwrited."
            warnings.warn(msg)
        else:  # pragma: no cover
            base_path.mkdir(parents=True, exist_ok=True)

        with open(base_path / "init_args.json", "w+") as file:
            json.dump(encoder_rule_dict, file)

    @classmethod
    def load(cls, path: str) -> "LabelEncodingRule":
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json", "r") as file:
            encoder_rule_dict = json.loads(file.read())

        string_column_type = encoder_rule_dict["fitted_args"]["column_type"]
        if "str" in string_column_type:
            column_type = str
        elif "int" in string_column_type:
            column_type = int
        elif "float" in string_column_type:
            column_type = float

        encoder_rule_dict["init_args"]["mapping"] = {
            column_type(key): int(value) for key, value in encoder_rule_dict["init_args"]["mapping"].items()
        }

        encoding_rule = cls(**encoder_rule_dict["init_args"])
        encoding_rule._target_col = encoder_rule_dict["fitted_args"]["target_col"]
        encoding_rule._is_fitted = encoder_rule_dict["fitted_args"]["is_fitted"]
        return encoding_rule


class SequenceEncodingRule(LabelEncodingRule):
    """
    Implementation of the encoding rule for grouped categorical variables of PySpark, Pandas and Polars Data Frames.
    Grouped means that one cell of the table contains a list with categorical values.
    Encodes target labels with value between 0 and n_classes-1 for the given column.
    It is recommended to use together with the LabelEncoder.
    """

    _FAKE_INDEX_COLUMN_NAME: str = "__index__"

    def fit(self, df: DataFrameLike) -> "SequenceEncodingRule":
        """
        Fits encoder to input dataframe.

        :param df: input dataframe.
        :returns: fitted EncodingRule.
        """
        if self._mapping is not None:
            return self

        if isinstance(df, PandasDataFrame):
            self._fit_pandas(df[[self.column]].explode(self.column))
        elif isinstance(df, SparkDataFrame):
            self._fit_spark(df.select(self.column).withColumn(self.column, sf.explode(self.column)))
        elif isinstance(df, PolarsDataFrame):
            self._fit_polars(df.select(self.column).explode(self.column))
        else:
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)
        self._inverse_mapping = self._make_inverse_mapping()
        self._inverse_mapping_list = self._make_inverse_mapping_list()
        if self._handle_unknown == "use_default_value" and self._default_value in self._inverse_mapping:
            msg = (
                "The used value for default_value "
                f"{self._default_value} is one of the "
                "values already used for encoding the "
                "seen labels."
            )
            raise ValueError(msg)
        self._is_fitted = True
        return self

    def partial_fit(self, df: DataFrameLike) -> "SequenceEncodingRule":
        """
        Fits new data to already fitted encoder.

        :param df: input dataframe.
        :returns: fitted EncodingRule.
        """
        if self._mapping is None:
            return self.fit(df)
        if isinstance(df, SparkDataFrame):
            self._partial_fit_spark(df.select(self.column).withColumn(self.column, sf.explode(self.column)))
        elif isinstance(df, PandasDataFrame):
            self._partial_fit_pandas(df[[self.column]].explode(self.column))
        elif isinstance(df, PolarsDataFrame):
            self._partial_fit_polars(df.select(self.column).explode(self.column))
        else:
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)

        self._is_fitted = True
        return self

    def _transform_spark(self, df: SparkDataFrame, default_value: Optional[int]) -> SparkDataFrame:
        map_expr = sf.create_map([sf.lit(x) for x in chain(*self.get_mapping().items())])
        encoded_df = df.withColumn(self._target_col, sf.transform(self.column, lambda x: map_expr.getItem(x)))

        if self._handle_unknown == "drop":
            encoded_df = encoded_df.withColumn(self._target_col, sf.filter(self._target_col, lambda x: x.isNotNull()))
            if encoded_df.select(sf.max(sf.size(self._target_col))).first()[0] == 0:
                warnings.warn(
                    f"You are trying to transform dataframe with all values are unknown for {self._col}, "
                    "with `handle_unknown_strategy=drop` leads to empty dataframe",
                    LabelEncoderTransformWarning,
                )
        elif self._handle_unknown == "error":
            if (
                encoded_df.select(sf.sum(sf.array_contains(self._target_col, -1).isNull().cast("integer"))).first()[0]
                != 0
            ):
                msg = f"Found unknown labels in column {self._col} during transform"
                raise ValueError(msg)
        else:
            if default_value:
                encoded_df = encoded_df.withColumn(
                    self._target_col,
                    sf.transform(self._target_col, lambda x: sf.when(x.isNull(), default_value).otherwise(x)),
                )

        result_df = encoded_df.drop(self._col).withColumnRenamed(self._target_col, self._col)
        return result_df

    def _transform_pandas(self, df: PandasDataFrame, default_value: Optional[int]) -> PandasDataFrame:
        mapping = self.get_mapping()
        joined_df = df.copy()
        if self._handle_unknown == "drop":
            max_array_len = 0

            def encode_func(array_col):
                nonlocal mapping, max_array_len
                res = []
                for x in array_col:
                    cur_len = 0
                    mapped = mapping.get(x)
                    if mapped is not None:
                        res.append(mapped)
                        cur_len += 1
                    max_array_len = max(max_array_len, cur_len)
                return res

            joined_df[self._target_col] = joined_df[self._col].apply(encode_func)
            if max_array_len == 0:
                warnings.warn(
                    f"You are trying to transform dataframe with all values are unknown for {self._col}, "
                    "with `handle_unknown_strategy=drop` leads to empty dataframe",
                    LabelEncoderTransformWarning,
                )
        elif self._handle_unknown == "error":
            none_count = 0

            def encode_func(array_col):
                nonlocal mapping, none_count
                res = []
                for x in array_col:
                    mapped = mapping.get(x)
                    if mapped is None:
                        none_count += 1
                    else:
                        res.append(mapped)
                return res

            joined_df[self._target_col] = joined_df[self._col].apply(encode_func)
            if none_count != 0:
                msg = f"Found unknown labels in column {self._col} during transform"
                raise ValueError(msg)
        else:

            def encode_func(array_col):
                nonlocal mapping
                return [mapping.get(x, default_value) for x in array_col]

            joined_df[self._target_col] = joined_df[self._col].apply(encode_func)

        result_df = joined_df.drop(self._col, axis=1).rename(columns={self._target_col: self._col})
        return result_df

    def _transform_polars(self, df: PolarsDataFrame, default_value: Optional[int]) -> SparkDataFrame:
        transformed_df = df.with_columns(
            pl.col(self._col)
            .list.eval(
                pl.element().replace_strict(
                    self.get_mapping(), default=default_value if self._handle_unknown == "use_default_value" else None
                ),
                parallel=True,
            )
            .alias(self._target_col)
        )
        if self._handle_unknown == "drop":
            transformed_df = transformed_df.with_columns(pl.col(self._target_col).list.drop_nulls())
            if (
                transformed_df.with_columns(pl.col(self._target_col).list.len()).select(pl.sum(self._target_col)).item()
                == 0
            ):
                warnings.warn(
                    f"You are trying to transform dataframe with all values are unknown for {self._col}, "
                    "with `handle_unknown_strategy=drop` leads to empty dataframe",
                    LabelEncoderTransformWarning,
                )
        elif self._handle_unknown == "error":
            none_checker = transformed_df.with_columns(
                pl.col(self._target_col).list.contains(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64)
            )
            if none_checker.select(pl.sum(self._target_col)).item() != 0:
                msg = f"Found unknown labels in column {self._col} during transform"
                raise ValueError(msg)

        result_df = transformed_df.drop(self._col).rename({self._target_col: self._col})
        return result_df

    def _inverse_transform_pandas(self, df: PandasDataFrame) -> PandasDataFrame:
        decoded_df = df.copy()

        def decode_func(array_col):
            return [self._inverse_mapping_list[x] for x in array_col]

        decoded_df[self._col] = decoded_df[self._col].apply(decode_func)
        return decoded_df

    def _inverse_transform_polars(self, df: PolarsDataFrame) -> PolarsDataFrame:
        mapping_size = len(self._inverse_mapping_list)
        transformed_df = df.with_columns(
            pl.col(self._col).list.eval(
                pl.element().replace_strict(old=list(range(mapping_size)), new=self._inverse_mapping_list),
                parallel=True,
            )
        )
        return transformed_df

    def _inverse_transform_spark(self, df: SparkDataFrame) -> SparkDataFrame:
        array_expr = sf.array([sf.lit(x) for x in self._inverse_mapping_list])
        decoded_df = df.withColumn(
            self._target_col, sf.transform(self._col, lambda x: sf.element_at(array_expr, x + 1))
        )
        return decoded_df.drop(self._col).withColumnRenamed(self._target_col, self._col)


class LabelEncoder:
    """
    Applies multiple label encoding rules to the data frame.

    >>> import pandas as pd
    >>> user_interactions = pd.DataFrame([
    ...     ("u1", "item_1", "item_1", [1, 2, 3]),
    ...     ("u2", "item_2", "item_2", [3, 4, 5]),
    ...     ("u3", "item_3", "item_3", [-1, -2, 4]),
    ... ], columns=["user_id", "item_1", "item_2", "list"])
    >>> user_interactions
        user_id	item_1	item_2	list
    0	u1	    item_1	item_1	[1, 2, 3]
    1	u2	    item_2	item_2	[3, 4, 5]
    2	u3	    item_3	item_3	[-1, -2, 4]
    >>> encoder = LabelEncoder([
    ...     LabelEncodingRule("user_id"),
    ...     LabelEncodingRule("item_1"),
    ...     LabelEncodingRule("item_2"),
    ...     SequenceEncodingRule("list"),
    ... ])
    >>> mapped_interactions = encoder.fit_transform(user_interactions)
    >>> mapped_interactions
       user_id  item_1  item_2  list
    0        0       0       0  [0, 1, 2]
    1        1       1       1  [2, 3, 4]
    2        2       2       2  [5, 6, 3]
    >>> encoder.mapping
    {'user_id': {'u1': 0, 'u2': 1, 'u3': 2},
    'item_1': {'item_1': 0, 'item_2': 1, 'item_3': 2},
    'item_2': {'item_1': 0, 'item_2': 1, 'item_3': 2},
    'list': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, -1: 5, -2: 6}}
    >>> encoder.inverse_mapping
    {'user_id': {0: 'u1', 1: 'u2', 2: 'u3'},
    'item_1': {0: 'item_1', 1: 'item_2', 2: 'item_3'},
    'item_2': {0: 'item_1', 1: 'item_2', 2: 'item_3'},
    'list': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: -1, 6: -2}}
    >>> new_encoder = LabelEncoder([
    ...    LabelEncodingRule("user_id", encoder.mapping["user_id"]),
    ...    LabelEncodingRule("item_1", encoder.mapping["item_1"]),
    ...    LabelEncodingRule("item_2", encoder.mapping["item_2"]),
    ...    SequenceEncodingRule("list", encoder.mapping["list"]),
    ... ])
    >>> new_encoder.inverse_transform(mapped_interactions)
      user_id item_1 item_2	list
    0      u1 item_1 item_1	[1, 2, 3]
    1      u2 item_2 item_2	[3, 4, 5]
    2      u3 item_3 item_3	[-1, -2, 4]
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
        columns = [i.column for i in self.rules]
        for column, handle_unknown in handle_unknown_rules.items():
            if column not in columns:
                msg = f"Column {column} not found."
                raise ValueError(msg)
            rule = list(filter(lambda x: x.column == column, self.rules))
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
        columns = [i.column for i in self.rules]
        for column, default_value in default_value_rules.items():
            if column not in columns:
                msg = f"Column {column} not found."
                raise ValueError(msg)
            rule = list(filter(lambda x: x.column == column, self.rules))
            rule[0].set_default_value(default_value)

    def save(
        self,
        path: str,
    ) -> None:
        encoder_dict = {}
        encoder_dict["_class_name"] = self.__class__.__name__

        base_path = Path(path).with_suffix(".replay").resolve()
        if os.path.exists(base_path):  # pragma: no cover
            msg = "There is already LabelEncoder object saved at the given path. File will be overwrited."
            warnings.warn(msg)
        else:  # pragma: no cover
            base_path.mkdir(parents=True, exist_ok=True)

        encoder_dict["rule_names"] = []

        for rule in self.rules:
            path_suffix = f"{rule.__class__.__name__}_{rule.column}"
            rule.save(str(base_path) + f"/rules/{path_suffix}")
            encoder_dict["rule_names"].append(path_suffix)

        with open(base_path / "init_args.json", "w+") as file:
            json.dump(encoder_dict, file)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json", "r") as file:
            encoder_dict = json.loads(file.read())
        rules = []
        for root, dirs, files in os.walk(str(base_path) + "/rules/"):
            for d in dirs:
                if d.split(".")[0] in encoder_dict["rule_names"]:
                    with open(root + d + "/init_args.json", "r") as file:
                        encoder_rule_dict = json.loads(file.read())
                    rules.append(globals()[encoder_rule_dict["_class_name"]].load(root + d))

        encoder = cls(rules=rules)
        return encoder
