from importlib.util import find_spec
from typing import Iterable, Union

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame


class MissingImportType:
    """
    Replacement class with missing import
    """

class FeatureUnavailableError(Exception):
    """Exception class for failing a conditional import check."""

class FeatureUnavailableWarning(Warning):
    """Warning class for failing a conditional import check."""


PYSPARK_AVAILABLE = find_spec("pyspark")
if PYSPARK_AVAILABLE:
    from pyspark.sql import DataFrame

    SparkDataFrame = DataFrame
else:
    SparkDataFrame = MissingImportType

TORCH_AVAILABLE = find_spec("torch")

DataFrameLike = Union[PandasDataFrame, SparkDataFrame, PolarsDataFrame]
IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]


# Conditional import flags 
ANN_AVAILABLE = all([
    find_spec("nmslib"),
    find_spec("hnswlib"),
    find_spec("pyarrow"),
])
OPENVINO_AVAILABLE = TORCH_AVAILABLE and find_spec("onnx") and find_spec("openvino")
OPTUNA_AVAILABLE = find_spec("optuna")
