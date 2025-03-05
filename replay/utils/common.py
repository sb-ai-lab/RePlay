import functools
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from polars import from_pandas as pl_from_pandas

from replay.data.dataset import Dataset
from replay.preprocessing import (
    LabelEncoder,
    LabelEncodingRule,
)
from replay.splitters import (
    ColdUserRandomSplitter,
    KFolds,
    LastNSplitter,
    NewUsersSplitter,
    RandomSplitter,
    RatioSplitter,
    TimeSplitter,
    TwoStageSplitter,
)
from replay.utils import (
    TORCH_AVAILABLE,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)
from replay.utils.pandas_utils import filter_cold as pandas_filter_cold
from replay.utils.polars_utils import filter_cold as polars_filter_cold
from replay.utils.spark_utils import (
    convert2spark as pandas_to_spark,
    filter_cold as spark_filter_cold,
    spark_to_pandas,
)
from replay.utils.types import DataFrameLike

SavableObject = Union[
    ColdUserRandomSplitter,
    KFolds,
    LastNSplitter,
    NewUsersSplitter,
    RandomSplitter,
    RatioSplitter,
    TimeSplitter,
    TwoStageSplitter,
    Dataset,
    LabelEncoder,
    LabelEncodingRule,
]

if TORCH_AVAILABLE:
    from replay.data.nn import PandasSequentialDataset, PolarsSequentialDataset, SequenceTokenizer

    SavableObject = Union[
        SavableObject,
        SequenceTokenizer,
        PandasSequentialDataset,
        PolarsSequentialDataset,
    ]


def save_to_replay(obj: SavableObject, path: Union[str, Path]) -> None:
    """
    General function to save RePlay models, splitters and tokenizer.

    :param path: Path to save the object.
    """
    obj.save(path)


def load_from_replay(path: Union[str, Path], **kwargs) -> SavableObject:
    """
    General function to load RePlay models, splitters and tokenizer.

    :param path: Path to save the object.
    """
    path = Path(path).with_suffix(".replay").resolve()
    with open(path / "init_args.json", "r") as file:
        class_name = json.loads(file.read())["_class_name"]
    obj_type = globals()[class_name]
    obj = obj_type.load(path, **kwargs)

    return obj


def _check_if_dataframe(var: Any):
    if not isinstance(var, (SparkDataFrame, PolarsDataFrame, PandasDataFrame)):
        msg = f"Object of type {type(var)} is not a dataframe of known type (can be pandas|spark|polars)"
        raise ValueError(msg)


def check_if_dataframe(*args_to_check: str) -> Callable[..., Any]:
    def decorator_func(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrap_func(*args: Any, **kwargs: Any) -> Any:
            extended_kwargs = {}
            extended_kwargs.update(kwargs)
            extended_kwargs.update(dict(zip(inspect.signature(func).parameters.keys(), args)))
            # add default param values to dict with arguments
            extended_kwargs.update(
                {
                    x.name: x.default
                    for x in inspect.signature(func).parameters.values()
                    if x.name not in extended_kwargs and x.default is not x.empty
                }
            )
            vals_to_check = [extended_kwargs[_arg] for _arg in args_to_check]
            for val in vals_to_check:
                _check_if_dataframe(val)
            return func(*args, **kwargs)

        return wrap_func

    return decorator_func


@check_if_dataframe("data")
def convert2pandas(
    data: Union[SparkDataFrame, PolarsDataFrame, PandasDataFrame], allow_collect_to_master: bool = False
) -> PandasDataFrame:
    """
    Convert the spark|polars DataFrame to a pandas.DataFrame.
    Returns unchanged dataframe if the input is already of type pandas.DataFrame.

    :param data: The dataframe to convert. Can be polars|spark|pandas DataFrame.
    :param allow_collect_to_master: If set to False (default) raises a warning
        about collecting parallelized data to the master node.
    """
    if isinstance(data, PandasDataFrame):
        return data
    if isinstance(data, PolarsDataFrame):
        return data.to_pandas()
    if isinstance(data, SparkDataFrame):
        return spark_to_pandas(data, allow_collect_to_master, from_constructor=False)


@check_if_dataframe("data")
def convert2polars(
    data: Union[SparkDataFrame, PolarsDataFrame, PandasDataFrame], allow_collect_to_master: bool = False
) -> PolarsDataFrame:
    """
    Convert the spark|pandas DataFrame to a polars.DataFrame.
    Returns unchanged dataframe if the input is already of type polars.DataFrame.

    :param data: The dataframe to convert. Can be spark|pandas|polars DataFrame.
    :param allow_collect_to_master: If set to False (default) raises a warning
        about collecting parallelized data to the master node.
    """
    if isinstance(data, PandasDataFrame):
        return pl_from_pandas(data)
    if isinstance(data, PolarsDataFrame):
        return data
    if isinstance(data, SparkDataFrame):
        return pl_from_pandas(spark_to_pandas(data, allow_collect_to_master, from_constructor=False))


@check_if_dataframe("data")
def convert2spark(data: Union[SparkDataFrame, PolarsDataFrame, PandasDataFrame]) -> SparkDataFrame:
    """
    Convert the pandas|polars DataFrame to a pysaprk.sql.DataFrame.
    Returns unchanged dataframe if the input is already of type pysaprk.sql.DataFrame.

    :param data: The dataframe to convert. Can be pandas|polars|spark Datarame.
    """
    if isinstance(data, (PandasDataFrame, SparkDataFrame)):
        return pandas_to_spark(data)
    if isinstance(data, PolarsDataFrame):
        return pandas_to_spark(data.to_pandas())


def filter_cold(
    df: Optional[DataFrameLike],
    warm_df: DataFrameLike,
    col_name: str,
) -> Tuple[int, Optional[DataFrameLike]]:
    return NotImplementedError()
    type_df1, type_df2 = type(df), type(warm_df)
    if type_df1 != type_df2:
        msg = f"Type of input attributes 'df' ({type_df1}) and 'warm_df' ({type_df2}) are not the same"
        raise ValueError(msg)
    if isinstance(df, SparkDataFrame):
        return spark_filter_cold(df, warm_df, col_name)
    elif isinstance(df, PandasDataFrame):
        return pandas_filter_cold(df, warm_df, col_name)
    elif isinstance(df, PolarsDataFrame):
        return polars_filter_cold(df, warm_df, col_name)
