from collections.abc import Iterable
from importlib.util import find_spec
from typing import TypeAlias

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame


class MissingImport:
    """
    Replacement class with missing import
    """


class FeatureUnavailableError(Exception):
    """Exception class for failing a conditional import check."""


class FeatureUnavailableWarning(Warning):
    """Warning class for failing a conditional import check."""


PYSPARK_AVAILABLE = find_spec("pyspark")
if not PYSPARK_AVAILABLE:
    SparkDataFrame: TypeAlias = MissingImport
else:
    from pyspark.sql import DataFrame

    SparkDataFrame: TypeAlias = DataFrame


TORCH_AVAILABLE = find_spec("torch") and find_spec("lightning")

DataFrameLike: TypeAlias = PandasDataFrame | SparkDataFrame | PolarsDataFrame
IntOrList: TypeAlias = Iterable[int] | int
NumType: TypeAlias = int | float


# Conditional import flags
ANN_AVAILABLE = all(
    [
        find_spec("nmslib"),
        find_spec("hnswlib"),
        find_spec("pyarrow"),
    ]
)
OPENVINO_AVAILABLE = TORCH_AVAILABLE and find_spec("onnx") and find_spec("openvino")
OPTUNA_AVAILABLE = find_spec("optuna")
LIGHTFM_AVAILABLE = find_spec("lightfm")
