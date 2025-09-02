from importlib.util import find_spec
from typing import Iterable, Union

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame


class MissingImportType:
    """
    Replacement class with missing import
    """

PYSPARK_AVAILABLE = find_spec("pyspark")
if PYSPARK_AVAILABLE:
    from pyspark.sql import DataFrame

    SparkDataFrame = DataFrame
else:
    SparkDataFrame = MissingImportType

TORCH_AVAILABLE = find_spec("torch")
OPENVINO_AVAILABLE = TORCH_AVAILABLE and find_spec("onnx") and find_spec("openvino")

DataFrameLike = Union[PandasDataFrame, SparkDataFrame, PolarsDataFrame]
IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]
