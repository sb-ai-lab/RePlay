from typing import Iterable, Union

from pandas import DataFrame as PandasDataFrame


# pylint: disable=too-few-public-methods
class MissingImportType:
    """
    Replacement class with missing import
    """


try:
    from pyspark.sql import DataFrame as SparkDataFrame

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = MissingImportType  # type: ignore

try:
    # pylint: disable=unused-import
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DataFrameLike = Union[PandasDataFrame, SparkDataFrame]
IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]
