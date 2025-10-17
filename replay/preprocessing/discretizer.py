import abc
import json
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from sklearn.preprocessing import KBinsDiscretizer

from replay.utils import (
    PYSPARK_AVAILABLE,
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)

if PYSPARK_AVAILABLE:  # pragma: no cover
    from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
    from pyspark.sql.functions import isnan

HandleInvalidStrategies = Literal["error", "skip", "keep"]


class BaseDiscretizingRule(abc.ABC):  # pragma: no cover
    """
    Interface of the discretizing rule
    """

    @property
    @abc.abstractmethod
    def column(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def n_bins(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(self, df: DataFrameLike) -> "BaseDiscretizingRule":
        raise NotImplementedError()

    @abc.abstractmethod
    def partial_fit(self, df: DataFrameLike) -> "BaseDiscretizingRule":
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, df: DataFrameLike) -> DataFrameLike:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_handle_invalid(self, handle_invalid: HandleInvalidStrategies) -> None:
        raise NotImplementedError()

    def fit_transform(self, df: DataFrameLike) -> DataFrameLike:
        return self.fit(df).transform(df)


class GreedyDiscretizingRule(BaseDiscretizingRule):
    """
    Implementation of the Discretizing rule for a column of PySpark, Polars and Pandas DataFrames.
    Discretizes column values according to the Greedy binning strategy:
    https://github.com/microsoft/LightGBM/blob/master/src/io/bin.cpp#L78
    It is recommended to use together with the Discretizer.
    """

    _HANDLE_INVALID_STRATEGIES = ("error", "skip", "keep")

    def __init__(
        self,
        column: str,
        n_bins: int,
        min_data_in_bin: int = 1,
        handle_invalid: HandleInvalidStrategies = "keep",
    ) -> None:
        """
        :param column: Name of the column to discretize.
        :param n_bins: Number of intervals where data will be binned.
        :param min_data_in_bin: Minimum number of samples in one bin.
        :param handle_invalid: handle_invalid rule.
            indicates how to process NaN in data.
            If ``skip`` - filter out rows with invalid values.
            If ``error`` - throw an error.
            If ``keep`` - keep invalid values in a special additional bucket with number = n_bins.
            Default ``keep``.
        """
        self._n_bins = n_bins
        self._col = column
        self._min_data_in_bin = min_data_in_bin
        self._bins = None
        self._is_fitted = False

        if handle_invalid not in self._HANDLE_INVALID_STRATEGIES:
            msg = f"handle_invalid should be either 'error' or 'skip' or 'keep', got {handle_invalid}."
            raise ValueError(msg)
        self._handle_invalid = handle_invalid

    @property
    def column(self) -> str:
        return self._col

    @property
    def n_bins(self) -> str:
        return self._n_bins

    def _greedy_bin_find(
        self,
        distinct_values: np.ndarray,
        counts: np.ndarray,
        num_distinct_values: int,
        max_bin: int,
        total_cnt: int,
        min_data_in_bin: int,
    ) -> list[float]:
        """
        Computes bound for bins.

        :param distinct_values: Array of unique values.
        :param counts: Number of samples corresponding to the every unique value.
        :param num_distinct_values: Number of unique value.
        :param max_bin: Maximum bin number.
        :param total_cnt: Total number of samples.
        :param min_data_in_bin: Minimum number of samples in one bin.
        """
        bin_upper_bound = []
        assert max_bin > 0

        if total_cnt < max_bin * min_data_in_bin:
            warn_msg = f"Expected at least {max_bin*min_data_in_bin} samples (n_bins*min_data_in_bin) \
= ({self._n_bins}*{min_data_in_bin}). Got {total_cnt}. The number of bins will be less in the result"
            warnings.warn(warn_msg)
        if num_distinct_values <= max_bin:
            cur_cnt_inbin = 0
            for i in range(num_distinct_values - 1):
                cur_cnt_inbin += counts[i]
                if cur_cnt_inbin >= min_data_in_bin:
                    bin_upper_bound.append((distinct_values[i] + distinct_values[i + 1]) / 2.0)
                    cur_cnt_inbin = 0

            cur_cnt_inbin += counts[num_distinct_values - 1]
            bin_upper_bound.append(float("Inf"))

        else:
            if min_data_in_bin > 0:
                max_bin = min(max_bin, total_cnt // min_data_in_bin)
                max_bin = max(max_bin, 1)
            mean_bin_size = total_cnt / max_bin
            rest_bin_cnt = max_bin
            rest_sample_cnt = total_cnt

            is_big_count_value = counts >= mean_bin_size
            rest_bin_cnt -= np.sum(is_big_count_value)
            rest_sample_cnt -= np.sum(counts[is_big_count_value])

            mean_bin_size = rest_sample_cnt / rest_bin_cnt
            upper_bounds = [float("Inf")] * max_bin
            lower_bounds = [float("Inf")] * max_bin

            bin_cnt = 0
            lower_bounds[bin_cnt] = distinct_values[0]
            cur_cnt_inbin = 0

            for i in range(num_distinct_values - 1):  # pragma: no cover
                if not is_big_count_value[i]:
                    rest_sample_cnt -= counts[i]

                cur_cnt_inbin += counts[i]

                if (
                    is_big_count_value[i]
                    or cur_cnt_inbin >= mean_bin_size
                    or (is_big_count_value[i + 1] and cur_cnt_inbin >= max(1.0, mean_bin_size * 0.5))
                ):
                    upper_bounds[bin_cnt] = distinct_values[i]
                    bin_cnt += 1
                    lower_bounds[bin_cnt] = distinct_values[i + 1]
                    if bin_cnt >= max_bin - 1:
                        break
                    cur_cnt_inbin = 0
                    if not is_big_count_value[i]:
                        rest_bin_cnt -= 1
                        mean_bin_size = rest_sample_cnt / rest_bin_cnt

            bin_upper_bound = [(upper_bounds[i] + lower_bounds[i + 1]) / 2.0 for i in range(bin_cnt - 1)]
            bin_upper_bound.append(float("Inf"))
        return bin_upper_bound

    def _fit_spark(self, df: SparkDataFrame) -> None:
        warn_msg = "DataFrame will be partially converted to the Pandas type during internal calculations in 'fit'"
        warnings.warn(warn_msg)
        value_counts = df.groupBy(self._col).count().orderBy(self._col).toPandas()
        bins = [-float("inf")]
        bins += self._greedy_bin_find(
            value_counts[self._col].values,
            value_counts["count"].values,
            value_counts.shape[0],
            self._n_bins + 1,
            df.count(),
            self._min_data_in_bin,
        )
        self._bins = bins

    def _fit_pandas(self, df: PandasDataFrame) -> None:
        vc = df[self._col].value_counts().sort_index()
        bins = self._greedy_bin_find(
            vc.index.values, vc.values, len(vc), self._n_bins + 1, vc.sum(), self._min_data_in_bin
        )
        self._bins = [-np.inf, *bins]

    def _fit_polars(self, df: PolarsDataFrame) -> None:
        warn_msg = "DataFrame will be converted to the Pandas type during internal calculations in 'fit'"
        warnings.warn(warn_msg)
        self._fit_pandas(df.to_pandas())

    def fit(self, df: DataFrameLike) -> "GreedyDiscretizingRule":
        """
        Fits Discretizing Rule to input dataframe.

        :param df: input dataframe.
        :returns: fitted DiscretizingRule.
        """
        if self._is_fitted:
            return self

        df = self._validate_input(df)

        if isinstance(df, PandasDataFrame):
            self._fit_pandas(df)
        elif isinstance(df, SparkDataFrame):
            self._fit_spark(df)
        else:
            self._fit_polars(df)

        self._is_fitted = True
        return self

    def partial_fit(self, df: DataFrameLike) -> "GreedyDiscretizingRule":
        """
        Fits new data to already fitted DiscretizingRule.

        :param df: input dataframe.
        :returns: fitted DiscretizingRule.
        """
        if not self._is_fitted:
            return self.fit(df)

        msg = f"{self.__class__.__name__} is not implemented for partial_fit yet."
        raise NotImplementedError(msg)

    def _transform_pandas(self, df: PandasDataFrame) -> PandasDataFrame:
        binned_column = np.digitize(df[self._col].values, self._bins)
        binned_column -= 1
        df_transformed = df.copy()
        df_transformed.loc[:, self._col] = binned_column
        return df_transformed

    def _transform_spark(self, df: SparkDataFrame) -> SparkDataFrame:
        target_col = self._col + "_discretized"
        bucketizer = Bucketizer(
            splits=self._bins, inputCol=self._col, outputCol=target_col, handleInvalid=self._handle_invalid
        )
        return bucketizer.transform(df).drop(self._col).withColumnRenamed(target_col, self._col)

    def _transform_polars(self, df: PolarsDataFrame) -> PolarsDataFrame:
        warn_msg = "DataFrame will be converted to the Pandas type during internal calculations in 'transform'"
        warnings.warn(warn_msg)
        return pl.from_pandas(self._transform_pandas(df.to_pandas()))

    def transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Transforms input dataframe with fitted DiscretizingRule.

        :param df: input dataframe.
        :returns: transformed dataframe.
        """
        if not self._is_fitted:
            msg = "Discretizer is not fitted"
            raise RuntimeError(msg)

        df = self._validate_input(df)

        if isinstance(df, PandasDataFrame):
            transformed_df = self._transform_pandas(df)
        elif isinstance(df, SparkDataFrame):
            transformed_df = self._transform_spark(df)
        else:
            transformed_df = self._transform_polars(df)
        return transformed_df

    def set_handle_invalid(self, handle_invalid: HandleInvalidStrategies) -> None:
        """
        Sets strategy to handle invalid values.

        :param handle_invalid: handle invalid strategy.
        """
        if handle_invalid not in self._HANDLE_INVALID_STRATEGIES:
            msg = f"handle_invalid should be either 'error' or 'skip' or 'keep', got {handle_invalid}."
            raise ValueError(msg)
        self._handle_invalid = handle_invalid

    def _validate_input(self, df: DataFrameLike) -> DataFrameLike:
        if isinstance(df, PandasDataFrame):
            df_val = df.copy()
            if (self._handle_invalid == "error") and (df_val[self._col].isna().sum() > 0):
                msg = "Data contains NaN. 'handle_invalid' param equals 'error'. \
Set 'keep' or 'skip' for processing NaN."
                raise ValueError(msg)
            if self._handle_invalid == "skip":
                df_val = df_val.dropna(subset=[self._col], axis=0)
            return df_val

        elif isinstance(df, SparkDataFrame):
            if (self._handle_invalid == "error") and (df.filter(isnan(df[self._col])).count() > 0):
                msg = "Data contains NaN. 'handle_invalid' param equals 'error'. \
Set 'keep' or 'skip' for processing NaN."
                raise ValueError(msg)
            return df

        elif isinstance(df, PolarsDataFrame):
            if (self._handle_invalid == "error") and (df[self._col].is_null().sum() > 0):
                msg = "Data contains NaN. 'handle_invalid' param equals 'error'. \
Set 'keep' or 'skip' for processing NaN."
                raise ValueError(msg)
            if self._handle_invalid == "skip":
                df = df.clone().fill_nan(None).drop_nulls(subset=[self._col])
            return df

        else:
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)

    def save(
        self,
        path: str,
    ) -> None:
        discretizer_rule_dict = {}
        discretizer_rule_dict["_class_name"] = self.__class__.__name__
        discretizer_rule_dict["init_args"] = {
            "n_bins": self._n_bins,
            "column": self._col,
            "min_data_in_bin": self._min_data_in_bin,
            "handle_invalid": self._handle_invalid,
        }
        discretizer_rule_dict["fitted_args"] = {
            "bins": self._bins,
            "is_fitted": self._is_fitted,
        }

        base_path = Path(path).with_suffix(".replay").resolve()

        if os.path.exists(base_path):  # pragma: no cover
            msg = "There is already DiscretizingRule object saved at the given path. File will be overwrited."
            warnings.warn(msg)
        else:  # pragma: no cover
            base_path.mkdir(parents=True, exist_ok=True)

        with open(base_path / "init_args.json", "w+") as file:
            json.dump(discretizer_rule_dict, file)

    @classmethod
    def load(cls, path: str) -> "GreedyDiscretizingRule":
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json") as file:
            discretizer_rule_dict = json.loads(file.read())

        discretizer_rule = cls(**discretizer_rule_dict["init_args"])
        discretizer_rule._bins = discretizer_rule_dict["fitted_args"]["bins"]
        discretizer_rule._is_fitted = discretizer_rule_dict["fitted_args"]["is_fitted"]
        return discretizer_rule


class QuantileDiscretizingRule(BaseDiscretizingRule):
    """
    Implementation of the Discretizing rule for a column of PySpark, Polars and Pandas DataFrames.
    Discretizes columns values according to the Quantile strategy. All the data will be distributed
    into buckets with approximately same size.
    It is recommended to use together with the Discretizer.
    """

    _HANDLE_INVALID_STRATEGIES = ("error", "skip", "keep")

    def __init__(
        self,
        column: str,
        n_bins: int,
        handle_invalid: HandleInvalidStrategies = "keep",
    ) -> None:
        """
        :param column: Name of the column to discretize.
        :param n_bins: Number of intervals where data will be binned.
        :param handle_invalid: handle_invalid rule.
            indicates how to process NaN in data.
            If ``skip`` - filter out rows with invalid values.
            If ``error`` - throw an error.
            If ``keep`` - keep invalid values in a special additional bucket with number = n_bins.
            Default ``keep``.
        """
        self._n_bins = n_bins
        self._col = column
        self._bins = None
        self._discretizer = None
        self._is_fitted = False

        if handle_invalid not in self._HANDLE_INVALID_STRATEGIES:
            msg = f"handle_invalid should be either 'error' or 'ski[]' or 'keep', got {handle_invalid}."
            raise ValueError(msg)
        self._handle_invalid = handle_invalid

    @property
    def column(self) -> str:
        return self._col

    @property
    def n_bins(self) -> str:
        return self._n_bins

    def _fit_spark(self, df: SparkDataFrame) -> None:
        discretizer = QuantileDiscretizer(
            numBuckets=self._n_bins, inputCol=self._col, handleInvalid=self._handle_invalid
        )
        self._discretizer = discretizer.fit(df)
        self._bins = self._discretizer.getSplits()

    def _fit_pandas(self, df: PandasDataFrame) -> None:
        discretizer = KBinsDiscretizer(n_bins=self._n_bins, encode="ordinal", strategy="quantile")
        if self._handle_invalid == "keep":
            self._discretizer = discretizer.fit(df.dropna(subset=[self._col], axis=0)[[self._col]])
        else:
            self._discretizer = discretizer.fit(df[[self._col]])
        self._bins = self._discretizer.bin_edges_[0].astype(float).tolist()
        self._bins[0] = -np.inf
        self._bins[-1] = np.inf

    def _fit_polars(self, df: PolarsDataFrame) -> None:
        warn_msg = "DataFrame will be converted to the Pandas type during internal calculations in 'fit'"
        warnings.warn(warn_msg)
        self._fit_pandas(df.to_pandas())

    def fit(self, df: DataFrameLike) -> "GreedyDiscretizingRule":
        """
        Fits DiscretizingRule to input dataframe.

        :param df: input dataframe.
        :returns: fitted DiscretizingRule.
        """
        if self._is_fitted:
            return self

        df = self._validate_input(df)

        if isinstance(df, PandasDataFrame):
            self._fit_pandas(df)
        elif isinstance(df, SparkDataFrame):
            self._fit_spark(df)
        else:
            self._fit_polars(df)

        self._is_fitted = True
        return self

    def partial_fit(self, df: DataFrameLike):
        """
        Fits new data to already fitted DiscretizingRule.

        :param df: input dataframe.
        :returns: fitted DiscretizingRule.
        """
        if not self._is_fitted:
            return self.fit(df)

        msg = f"{self.__class__.__name__} is not implemented for partial_fit yet."
        raise NotImplementedError(msg)

    def _transform_pandas(self, df: PandasDataFrame) -> PandasDataFrame:
        df_nan_part = df[df[self._col].isna()]
        df_real_part = df[~df[self._col].isna()]

        binned_column = np.digitize(df_real_part[self._col].values, self._bins)
        binned_column -= 1

        df_transformed = df.copy()
        df_transformed.loc[df_real_part.index, self._col] = binned_column
        df_transformed.loc[df_nan_part.index, self._col] = [self._n_bins] * len(df_nan_part)
        return df_transformed

    def _transform_spark(self, df: SparkDataFrame) -> SparkDataFrame:
        target_col = self._col + "_discretized"
        bucketizer = Bucketizer(
            splits=self._bins, inputCol=self._col, outputCol=target_col, handleInvalid=self._handle_invalid
        )
        return bucketizer.transform(df).drop(self._col).withColumnRenamed(target_col, self._col)

    def _transform_polars(self, df: PolarsDataFrame) -> SparkDataFrame:
        warn_msg = "DataFrame will be converted to the Pandas type during internal calculations in 'transform'"
        warnings.warn(warn_msg)
        return pl.from_pandas(self._transform_pandas(df.to_pandas()))

    def transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Transforms input dataframe with fitted DiscretizingRule.

        :param df: input dataframe.
        :returns: transformed dataframe.
        """
        if not self._is_fitted:
            msg = "Discretizer is not fitted"
            raise RuntimeError(msg)

        df = self._validate_input(df)

        if isinstance(df, PandasDataFrame):
            transformed_df = self._transform_pandas(df)
        elif isinstance(df, SparkDataFrame):
            transformed_df = self._transform_spark(df)
        else:
            transformed_df = self._transform_polars(df)
        return transformed_df

    def set_handle_invalid(self, handle_invalid: HandleInvalidStrategies) -> None:
        """
        Sets strategy to handle invalid values.

        :param handle_invalid: handle invalid strategy.
        """
        if handle_invalid not in self._HANDLE_INVALID_STRATEGIES:
            msg = f"handle_invalid should be either 'error' or 'skip' or 'keep', got {handle_invalid}."
            raise ValueError(msg)
        self._handle_invalid = handle_invalid

    def _validate_input(self, df: DataFrameLike) -> DataFrameLike:
        if isinstance(df, PandasDataFrame):
            if (self._handle_invalid == "error") and (df[self._col].isna().sum() > 0):
                msg = "Data contains NaN. 'handle_invalid' param equals 'error'. \
Set 'keep' or 'skip' for processing NaN."
                raise ValueError(msg)
            if self._handle_invalid == "skip":
                df = df.copy().dropna(subset=[self._col], axis=0)

        elif isinstance(df, SparkDataFrame):
            if (self._handle_invalid == "error") and (df.filter(isnan(df[self._col])).count() > 0):
                msg = "Data contains NaN. 'handle_invalid' param equals 'error'. \
Set 'keep' or 'skip' for processing NaN."
                raise ValueError(msg)
            if self._handle_invalid == "skip":
                df = df.dropna(subset=[self._col])

        elif isinstance(df, PolarsDataFrame):
            if (self._handle_invalid == "error") and (df[self._col].is_null().sum() > 0):
                msg = "Data contains NaN. 'handle_invalid' param equals 'error'. \
Set 'keep' or 'skip' for processing NaN."
                raise ValueError(msg)
            if self._handle_invalid == "skip":
                df = df.clone().fill_nan(None).drop_nulls(subset=[self._col])

        else:
            msg = f"{self.__class__.__name__} is not implemented for {type(df)}"
            raise NotImplementedError(msg)
        return df

    def save(
        self,
        path: str,
    ) -> None:
        discretizer_rule_dict = {}
        discretizer_rule_dict["_class_name"] = self.__class__.__name__
        discretizer_rule_dict["init_args"] = {
            "n_bins": self._n_bins,
            "column": self._col,
            "handle_invalid": self._handle_invalid,
        }
        discretizer_rule_dict["fitted_args"] = {
            "bins": self._bins,
            "is_fitted": self._is_fitted,
        }

        base_path = Path(path).with_suffix(".replay").resolve()

        if os.path.exists(base_path):  # pragma: no cover
            msg = "There is already DiscretizingRule object saved at the given path. File will be overwrited."
            warnings.warn(msg)
        else:  # pragma: no cover
            base_path.mkdir(parents=True, exist_ok=True)

        with open(base_path / "init_args.json", "w+") as file:
            json.dump(discretizer_rule_dict, file)

    @classmethod
    def load(cls, path: str) -> "QuantileDiscretizingRule":
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json") as file:
            discretizer_rule_dict = json.loads(file.read())

        discretizer_rule = cls(**discretizer_rule_dict["init_args"])
        discretizer_rule._bins = discretizer_rule_dict["fitted_args"]["bins"]
        discretizer_rule._is_fitted = discretizer_rule_dict["fitted_args"]["is_fitted"]
        return discretizer_rule


class Discretizer:
    """
    Applies multiple discretizing rules to the data frame.
    Every sample will be distributed into bucket with number from the set [0, 1, ..., n_bins-1].
    """

    def __init__(self, rules: Sequence[BaseDiscretizingRule]):
        """
        :param rules: Sequence of rules.
        """
        self.rules = rules

    def fit(self, df: DataFrameLike) -> "Discretizer":
        """
        Fits a Discretizer by the input dataframe with given rules.

        :param df: input dataframe.
        :returns: fitted Discretizer.
        """
        for rule in self.rules:
            rule.fit(df)
        return self

    def partial_fit(self, df: DataFrameLike) -> "Discretizer":
        """
        Fits an already fitted Discretizer by the new input data frame with given rules.
        If Discretizer has not been fitted yet - performs default fit.

        :param df: input dataframe.
        :returns: fitted Discretizer.
        """
        for rule in self.rules:
            rule.partial_fit(df)
        return self

    def transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Transforms the input data frame.
        If the input data frame contains NaN values then they will be transformed by handle_invalid strategy.

        :param df: input dataframe.
        :returns: transformed dataframe.
        """
        for rule in self.rules:
            df = rule.transform(df)
        return df

    def fit_transform(self, df: DataFrameLike) -> DataFrameLike:
        """
        Fits a Discretizer by the input dataframe with given rules and transforms the input dataframe.

        :param df: input dataframe.
        :returns: transformed dataframe.
        """
        return self.fit(df).transform(df)

    def set_handle_invalid(self, handle_invalid_rules: dict[str, HandleInvalidStrategies]) -> None:
        """
        Modify handle_invalid strategy on already fitted Discretizer.

        :param handle_invalid_rules: handle_invalid rule.

        Example: {"item_id" : "keep", "user_id" : "skip", "category_column" : "error"}

        Default value examples:
            If ``skip`` - filter out rows with invalid values.
            If ``error`` - throw an error.
            If ``keep`` - keep invalid values in a special additional bucket with number = n_bins.
            Default ``keep``.
        """
        columns = [i.column for i in self.rules]
        for column, handle_invalid in handle_invalid_rules.items():
            if column not in columns:
                msg = f"Column {column} not found."
                raise ValueError(msg)
            rule = list(filter(lambda x: x.column == column, self.rules))
            rule[0].set_handle_invalid(handle_invalid)

    def save(
        self,
        path: str,
    ) -> None:
        discretizer_dict = {}
        discretizer_dict["_class_name"] = self.__class__.__name__

        base_path = Path(path).with_suffix(".replay").resolve()
        if os.path.exists(base_path):  # pragma: no cover
            msg = "There is already LabelEncoder object saved at the given path. File will be overwrited."
            warnings.warn(msg)
        else:  # pragma: no cover
            base_path.mkdir(parents=True, exist_ok=True)

        discretizer_dict["rule_names"] = []

        for rule in self.rules:
            path_suffix = f"{rule.__class__.__name__}_{rule.column}"
            rule.save(str(base_path) + f"/rules/{path_suffix}")
            discretizer_dict["rule_names"].append(path_suffix)

        with open(base_path / "init_args.json", "w+") as file:
            json.dump(discretizer_dict, file)

    @classmethod
    def load(cls, path: str) -> "Discretizer":
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json") as file:
            discretizer_dict = json.loads(file.read())
        rules = []
        for root, dirs, files in os.walk(str(base_path) + "/rules/"):
            for d in dirs:
                if d.split(".")[0] in discretizer_dict["rule_names"]:
                    with open(root + d + "/init_args.json") as file:
                        discretizer_rule_dict = json.loads(file.read())
                    rules.append(globals()[discretizer_rule_dict["_class_name"]].load(root + d))

        discretizer = cls(rules=rules)
        return discretizer
