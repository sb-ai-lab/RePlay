# pylint: disable=wildcard-import,invalid-name,unused-wildcard-import,unspecified-encoding
import os
import json
import shutil
from inspect import getfullargspec

import joblib
from os.path import exists, join

from pyspark.ml.feature import StringIndexerModel, IndexToString

from replay.data_preparator import Indexer
from replay.models import *
from replay.models.base_rec import BaseRecommender
from replay.session_handler import State


def save(model: BaseRecommender, path: str):
    """
    Save fitted model to disk as a folder

    :param model: Trained recommender
    :param path: destination where model files will be stored
    :return:
    """
    if exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
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


def save_indexer(indexer: Indexer, path: str):
    """
    Save fitted indexer to disk as a folder

    :param indexer: Trained indexer
    :param path: destination where indexer files will be stored
    :return:
    """
    if exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    init_args = indexer._init_args
    with open(join(path, "init_args.json"), "w") as json_file:
        json.dump(init_args, json_file)

    indexer.user_indexer.save(join(path, "user_indexer"))
    indexer.item_indexer.save(join(path, "item_indexer"))
    indexer.inv_user_indexer.save(join(path, "inv_user_indexer"))
    indexer.inv_item_indexer.save(join(path, "inv_item_indexer"))


def load_indexer(path: str):
    """
    Load saved indexer from disk

    :param path: path to folder
    :return: Restored trained model
    """
    State()
    with open(join(path, "init_args.json"), "r") as json_file:
        args = json.load(json_file)

    indexer = Indexer(**args)

    indexer.user_indexer = StringIndexerModel.load(join(path, "user_indexer"))
    indexer.item_indexer = StringIndexerModel.load(join(path, "item_indexer"))
    indexer.inv_user_indexer = IndexToString.load(
        join(path, "inv_user_indexer")
    )
    indexer.inv_item_indexer = IndexToString.load(
        join(path, "inv_item_indexer")
    )

    return indexer


def load(path: str):
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
