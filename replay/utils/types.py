from typing import Iterable, Union

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame


class MissingImportType:
    """
    Replacement class with missing import
    """


try:
    from pyspark.sql import DataFrame as SparkDataFrame

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = MissingImportType

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx  # noqa: F401
    import openvino  # noqa: F401

    OPENVINO_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    OPENVINO_AVAILABLE = False

DataFrameLike = Union[PandasDataFrame, SparkDataFrame, PolarsDataFrame]
IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]
