# pylint: disable=wildcard-import,invalid-name,unused-wildcard-import,unspecified-encoding
import os
import json
import shutil
from inspect import getfullargspec

from pyspark.ml.feature import StringIndexerModel, IndexToString
from os.path import exists, join

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

    model.user_indexer.save(join(path, "user_indexer"))
    model.item_indexer.save(join(path, "item_indexer"))
    model.inv_user_indexer.save(join(path, "inv_user_indexer"))
    model.inv_item_indexer.save(join(path, "inv_item_indexer"))

    dataframes = model._dataframes
    df_path = join(path, "dataframes")
    os.makedirs(df_path)
    for name, df in dataframes.items():
        df.write.parquet(join(df_path, name))


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

    model.user_indexer = StringIndexerModel.load(join(path, "user_indexer"))
    model.item_indexer = StringIndexerModel.load(join(path, "item_indexer"))
    model.inv_user_indexer = IndexToString.load(join(path, "inv_user_indexer"))
    model.inv_item_indexer = IndexToString.load(join(path, "inv_item_indexer"))

    df_path = join(path, "dataframes")
    dataframes = os.listdir(df_path)
    for name in dataframes:
        df = spark.read.parquet(join(df_path, name))
        setattr(model, name, df)

    model._load_model(join(path, "model"))
    return model
