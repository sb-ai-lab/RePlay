import json
from inspect import getfullargspec
from os.path import join
from pathlib import Path
from typing import Union

from replay.experimental.models.base_rec import BaseRecommender
from replay.experimental.preprocessing import Indexer
from replay.utils import PYSPARK_AVAILABLE
from replay.utils.session_handler import State
from replay.utils.spark_utils import load_pickled_from_parquet, save_picklable_to_parquet

if PYSPARK_AVAILABLE:
    import pyspark.sql.types as st
    from pyspark.ml.feature import IndexToString, StringIndexerModel
    from pyspark.sql import SparkSession

    from replay.utils.model_handler import get_fs

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

    if hasattr(model, "fit_users"):
        model.fit_users.write.mode("overwrite").parquet(join(df_path, "fit_users"))
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
    if name == "CQL":
        for a in args:
            if isinstance(args[a], dict) and "type" in args[a] and args[a]["type"] == "none":
                args[a]["params"] = {}
    init_args = getfullargspec(model_class.__init__).args
    init_args.remove("self")
    extra_args = set(args) - set(init_args)
    if len(extra_args) > 0:
        extra_args = {key: args[key] for key in args}
        init_args = {key: args[key] for key in init_args}
    else:
        init_args = args
        extra_args = {}

    model = model_class(**init_args)
    for arg in extra_args:
        model.arg = extra_args[arg]

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


def save_indexer(indexer: Indexer, path: Union[str, Path], overwrite: bool = False):
    """
    Save fitted indexer to disk as a folder

    :param indexer: Trained indexer
    :param path: destination where indexer files will be stored
    """
    if isinstance(path, Path):
        path = str(path)

    spark = State().session

    if not overwrite:
        fs = get_fs(spark)
        is_exists = fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))
        if is_exists:
            msg = f"Path '{path}' already exists. Mode is 'overwrite = False'."
            raise FileExistsError(msg)

    init_args = indexer._init_args
    init_args["user_type"] = str(indexer.user_type)
    init_args["item_type"] = str(indexer.item_type)
    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    df.coalesce(1).write.mode("overwrite").json(join(path, "init_args.json"))

    indexer.user_indexer.write().overwrite().save(join(path, "user_indexer"))
    indexer.item_indexer.write().overwrite().save(join(path, "item_indexer"))
    indexer.inv_user_indexer.write().overwrite().save(join(path, "inv_user_indexer"))
    indexer.inv_item_indexer.write().overwrite().save(join(path, "inv_item_indexer"))


def load_indexer(path: str) -> Indexer:
    """
    Load saved indexer from disk

    :param path: path to folder
    :return: restored Indexer
    """
    spark = State().session
    args = spark.read.json(join(path, "init_args.json")).first().asDict()

    user_type = args["user_type"]
    del args["user_type"]
    item_type = args["item_type"]
    del args["item_type"]

    indexer = Indexer(**args)

    if user_type.endswith("()"):
        user_type = user_type[:-2]
        item_type = item_type[:-2]
    user_type = getattr(st, user_type)
    item_type = getattr(st, item_type)
    indexer.user_type = user_type()
    indexer.item_type = item_type()

    indexer.user_indexer = StringIndexerModel.load(join(path, "user_indexer"))
    indexer.item_indexer = StringIndexerModel.load(join(path, "item_indexer"))
    indexer.inv_user_indexer = IndexToString.load(join(path, "inv_user_indexer"))
    indexer.inv_item_indexer = IndexToString.load(join(path, "inv_item_indexer"))

    return indexer
