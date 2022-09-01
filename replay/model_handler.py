# pylint: disable=wildcard-import,invalid-name,unused-wildcard-import,unspecified-encoding
import os
import json
import shutil
from inspect import getfullargspec

import joblib
from os.path import exists, join

import pyspark.sql.types as st
from pyspark.ml.feature import StringIndexerModel, IndexToString

from replay.data_preparator import Indexer
from replay.models import *
from replay.models.base_rec import BaseRecommender
from replay.session_handler import State
from replay.splitters import *


def prepare_dir(path):
    """
    Create empty `path` dir
    """
    if exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def save(model: BaseRecommender, path: str):
    """
    Save fitted model to disk as a folder

    :param model: Trained recommender
    :param path: destination where model files will be stored
    :return:
    """
    prepare_dir(path)
    model._save_model(join(path, "model"))

    init_args = model._init_args
    init_args["_model_name"] = str(model)
    with open(join(path, "init_args.json"), "w") as json_file:
        json.dump(init_args, json_file)

    dataframes = model._dataframes
    df_path = join(path, "dataframes")
    os.makedirs(df_path)
    for name, df in dataframes.items():
        df.write.parquet(join(df_path, name))
    model.fit_users.write.parquet(join(df_path, "fit_users"))
    model.fit_items.write.parquet(join(df_path, "fit_items"))

    joblib.dump(model.study, join(path, "study"))


def load(path: str) -> BaseRecommender:
    """
    Load saved model from disk

    :param path: path to model folder
    :return: Restored trained model
    """
    spark = State().session
    with open(join(path, "init_args.json"), "r") as json_file:
        args = json.load(json_file)
    name = args["_model_name"]
    del args["_model_name"]

    model_class = globals()[name]
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

    df_path = join(path, "dataframes")
    dataframes = os.listdir(df_path)
    for name in dataframes:
        df = spark.read.parquet(join(df_path, name))
        setattr(model, name, df)

    model._load_model(join(path, "model"))
    model.study = joblib.load(join(path, "study"))
    return model


def save_indexer(indexer: Indexer, path: str):
    """
    Save fitted indexer to disk as a folder

    :param indexer: Trained indexer
    :param path: destination where indexer files will be stored
    """
    prepare_dir(path)

    init_args = indexer._init_args
    init_args["user_type"] = str(indexer.user_type)
    init_args["item_type"] = str(indexer.item_type)
    with open(join(path, "init_args.json"), "w") as json_file:
        json.dump(init_args, json_file)

    indexer.user_indexer.save(join(path, "user_indexer"))
    indexer.item_indexer.save(join(path, "item_indexer"))
    indexer.inv_user_indexer.save(join(path, "inv_user_indexer"))
    indexer.inv_item_indexer.save(join(path, "inv_item_indexer"))


def load_indexer(path: str) -> Indexer:
    """
    Load saved indexer from disk

    :param path: path to folder
    :return: restored Indexer
    """
    State()
    with open(join(path, "init_args.json"), "r") as json_file:
        args = json.load(json_file)

    user_type = args["user_type"]
    del args["user_type"]
    item_type = args["item_type"]
    del args["item_type"]

    indexer = Indexer(**args)

    indexer.user_type = getattr(st, user_type)()
    indexer.item_type = getattr(st, item_type)()

    indexer.user_indexer = StringIndexerModel.load(join(path, "user_indexer"))
    indexer.item_indexer = StringIndexerModel.load(join(path, "item_indexer"))
    indexer.inv_user_indexer = IndexToString.load(
        join(path, "inv_user_indexer")
    )
    indexer.inv_item_indexer = IndexToString.load(
        join(path, "inv_item_indexer")
    )

    return indexer


def save_splitter(splitter: Splitter, path: str):
    """
    Save initialized splitter

    :param splitter: Initialized splitter
    :param path: destination where splitter files will be stored
    """
    prepare_dir(path)
    init_args = splitter._init_args
    init_args["_splitter_name"] = str(splitter)
    with open(join(path, "init_args.json"), "w") as json_file:
        json.dump(init_args, json_file)


def load_splitter(path: str) -> Splitter:
    """
    Load splitter

    :param path: path to folder
    :return: restored Splitter
    """
    State()
    with open(join(path, "init_args.json"), "r") as json_file:
        args = json.load(json_file)
    name = args["_splitter_name"]
    del args["_splitter_name"]
    splitter = globals()[name]
    return splitter(**args)
