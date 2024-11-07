import json
from pathlib import Path
from typing import Union

from polars import from_pandas as pl_from_pandas

from replay.data.dataset import Dataset
from replay.models import (
    KLUCB,
    SLIM,
    UCB,
    ALSWrap,
    AssociationRulesItemRec,
    CatPopRec,
    ClusterRec,
    ItemKNN,
    PopRec,
    QueryPopRec,
    RandomRec,
    ThompsonSampling,
    Wilson,
    Word2VecRec,
)
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
from replay.utils.decorators import check_if_dataframe
from replay.utils.spark_utils import (
    convert2spark as pandas_to_spark,
    spark_to_pandas,
)

SavableObject = Union[
    KLUCB,
    SLIM,
    UCB,
    ALSWrap,
    AssociationRulesItemRec,
    CatPopRec,
    ClusterRec,
    ItemKNN,
    PopRec,
    QueryPopRec,
    RandomRec,
    ThompsonSampling,
    Wilson,
    Word2VecRec,
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
