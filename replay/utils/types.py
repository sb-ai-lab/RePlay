# pylint: skip-file
from typing import Union

from pandas import DataFrame as PandasDataFrame


class MissingImportType:
    pass


try:
    from pyspark.sql import DataFrame as SparkDataFrame

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = MissingImportType  # type: ignore

DataFrameLike = Union[PandasDataFrame, SparkDataFrame]
