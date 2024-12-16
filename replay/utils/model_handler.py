import functools
import json
import os
import pickle
import warnings
from os.path import join
from pathlib import Path
from typing import Any, Callable, Optional, Union

from replay.data.dataset_utils import DatasetLabelEncoder
from replay.models import *
from replay.models.base_rec import BaseRecommender
from replay.splitters import *

from .session_handler import State
from .types import PYSPARK_AVAILABLE

if PYSPARK_AVAILABLE:
    from pyspark.sql import SparkSession

    from .spark_utils import load_pickled_from_parquet, save_picklable_to_parquet

    def get_fs(spark: SparkSession):
        """
        Gets `org.apache.hadoop.fs.FileSystem` instance from JVM gateway

        :param spark: spark session
        :return:
        """
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        return fs

    def get_list_of_paths(spark: SparkSession, dir_path: str):
        """
        Returns list of paths to files in the `dir_path`

        :param spark: spark session
        :param dir_path: path to dir in hdfs or local disk
        :return: list of paths to files
        """
        fs = get_fs(spark)
        statuses = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(dir_path))
        return [str(f.getPath()) for f in statuses]


def save(model: BaseRecommender, path: Union[str, Path], overwrite: bool = False):
    """
    Save fitted model to disk as a folder

    :param model: Trained recommender
    :param path: destination where model files will be stored
    :return:
    """
    if isinstance(path, Path):
        path = str(path)

    spark = State().session

    fs = get_fs(spark)
    if not overwrite:
        is_exists = fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))
        if is_exists:
            msg = f"Path '{path}' already exists. Mode is 'overwrite = False'."
            raise FileExistsError(msg)

    fs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(path))
    model._save_model(join(path, "model"))

    init_args = model._init_args
    init_args["_model_name"] = str(model)
    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    df.coalesce(1).write.mode("overwrite").option("ignoreNullFields", "false").json(join(path, "init_args.json"))

    dataframes = model._dataframes
    df_path = join(path, "dataframes")
    for name, df in dataframes.items():
        if df is not None:
            df.write.mode("overwrite").parquet(join(df_path, name))

    if hasattr(model, "fit_queries"):
        model.fit_queries.write.mode("overwrite").parquet(join(df_path, "fit_queries"))
    if hasattr(model, "fit_items"):
        model.fit_items.write.mode("overwrite").parquet(join(df_path, "fit_items"))
    if hasattr(model, "study"):
        save_picklable_to_parquet(model.study, join(path, "study"))


def load(path: str, model_type=None) -> BaseRecommender:
    """
    Load saved model from disk

    :param path: path to model folder
    :return: Restored trained model
    """
    spark = State().session
    args = spark.read.json(join(path, "init_args.json")).first().asDict(recursive=True)
    name = args["_model_name"]
    del args["_model_name"]

    model_class = model_type if model_type is not None else globals()[name]

    model = model_class(**args)

    dataframes_paths = get_list_of_paths(spark, join(path, "dataframes"))
    for dataframe_path in dataframes_paths:
        df = spark.read.parquet(dataframe_path)
        attr_name = dataframe_path.split("/")[-1]
        setattr(model, attr_name, df)

    model._load_model(join(path, "model"))
    fs = get_fs(spark)
    model.study = (
        load_pickled_from_parquet(join(path, "study"))
        if fs.exists(spark._jvm.org.apache.hadoop.fs.Path(join(path, "study")))
        else None
    )

    return model


def save_encoder(encoder: DatasetLabelEncoder, path: Union[str, Path]) -> None:
    """
    Save fitted DatasetLabelEncoder to disk as a folder

    :param encoder: Trained encoder
    :param path: destination where encoder files will be stored
    """
    if isinstance(path, Path):
        path = str(path)

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "encoder.pickle"), "wb") as cached_file:
        pickle.dump(encoder, cached_file)


def load_encoder(path: Union[str, Path]) -> DatasetLabelEncoder:
    """
    Load saved encoder from disk

    :param path: path to folder
    :return: restored DatasetLabelEncoder
    """
    if isinstance(path, Path):
        path = str(path)

    with open(os.path.join(path, "encoder.pickle"), "rb") as f:
        encoder = pickle.load(f)

    return encoder


def save_splitter(splitter: Splitter, path: str, overwrite: bool = False):
    """
    Save initialized splitter

    :param splitter: Initialized splitter
    :param path: destination where splitter files will be stored
    """
    init_args = splitter._init_args
    init_args["_splitter_name"] = str(splitter)
    spark = State().session
    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    if overwrite:
        df.coalesce(1).write.mode("overwrite").json(join(path, "init_args.json"))
    else:
        df.coalesce(1).write.json(join(path, "init_args.json"))


def load_splitter(path: str) -> Splitter:
    """
    Load splitter

    :param path: path to folder
    :return: restored Splitter
    """
    spark = State().session
    args = spark.read.json(join(path, "init_args.json")).first().asDict()
    name = args["_splitter_name"]
    del args["_splitter_name"]
    splitter = globals()[name]
    return splitter(**args)


def deprecation_warning(message: Optional[str] = None) -> Callable[..., Any]:
    """
    Decorator that throws deprecation warnings.

    :param message: message to deprecation warning without func name.
    """
    base_msg = "will be deprecated in future versions."

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg = f"{func.__qualname__} {message if message else base_msg}"
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return wrapper

    return decorator
