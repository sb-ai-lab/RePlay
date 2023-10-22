from typing import Iterable, Union

import pandas as pd
from pyspark.sql import DataFrame


IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]
AnyDataFrame = Union[DataFrame, pd.DataFrame]
