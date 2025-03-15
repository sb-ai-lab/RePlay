import json
import os
import pickle
from os.path import join
from pathlib import Path
from typing import Union

from replay.data.dataset_utils import DatasetLabelEncoder
from replay.models import (
    KLUCB,  # noqa: F401
    SLIM,  # noqa: F401
    UCB,  # noqa: F401
    ALSWrap,  # noqa: F401
    AssociationRulesItemRec,  # noqa: F401
    CatPopRec,  # noqa: F401
    ClusterRec,  # noqa: F401
    ItemKNN,  # noqa: F401
    LinUCB,  # noqa: F401
    PopRec,  # noqa: F401
    QueryPopRec,  # noqa: F401
    RandomRec,  # noqa: F401
    ThompsonSampling,  # noqa: F401
    Wilson,  # noqa: F401
    Word2VecRec,  # noqa: F401
    _BaseRecommenderSparkImpl,
    client_model_list,
)
from replay.models.base_rec_client import BaseRecommenderClient
from replay.models.implementations import *
from replay.splitters import *
from replay.utils.warnings import deprecation_warning  # noqa: F401

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


def save(
    model: Union[BaseRecommenderClient, _BaseRecommenderSparkImpl], path: Union[str, Path], overwrite: bool = False
):
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
    is_many_frameworks_model = type(model) in client_model_list
    fs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(path))
    if is_many_frameworks_model and model._impl is not None:
        if model._impl is not None:
            model = model._impl
        else:
            msg = "Cant save client of model. Fit model, to give client his implementation"
            raise ValueError(msg)

    model._save_model(join(path, "model"))

    model._save_model(join(path, "model"))
    init_args = model._init_args
    init_args["_model_name"] = str(model)
    dataframes = model._dataframes

    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    df.coalesce(1).write.mode("overwrite").option("ignoreNullFields", "false").json(join(path, "init_args.json"))
    # TODO: let's use repartition(1) instead of coalesce - its faster and more optimize

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


def load(path: str, model_type=None) -> _BaseRecommenderSparkImpl:
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
    if name in list(map(str, implementations_list)):
        client_class = globals()[
            name.replace("Spark", "").replace("Polars", "").replace("Pandas", "")
        ]  # TODO доработать передачу переменной model_type
        client = client_class(**args)
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
    if name in list(map(str, implementations_list)):
        client._impl = model
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
