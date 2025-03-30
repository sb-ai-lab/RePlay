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


def save(  # TODO: Think how to save model's implementations on pandas and polars without spark
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
    spark = State().session  # TODO: Think how to load model's implementations on pandas and polars without spark
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


'''
def save(
    model: Union[BaseRecommenderClient, _BaseRecommenderSparkImpl], path: Union[str, Path], overwrite: bool = False
):
    """
    Save fitted model to disk as a folder

    :param model: Trained recommender
    :param path: destination where model files will be stored
    :return:
    """
    try:
        _save_spark(model, path, overwrite)
    except:
        _save_core(model, path, overwrite)


def _save_core(model: BaseRecommenderClient, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if not overwrite and path.exists():
        msg = f"Path '{path}' already exists. Mode is 'overwrite = False'."
        raise FileExistsError(msg)
    path.mkdir(parents=True, exist_ok=True)
    model_save_path = path / "model"
    model_save_path.mkdir(exist_ok=True)
    model._save_model(str(model_save_path))
    is_many_frameworks_model = type(model) in client_model_list
    if is_many_frameworks_model and model.is_fitted:
        client_params = {}
        client_params["model_class"] = str(model)
        client_params["impl_class"] = str(model._impl)
        for attr in model.attributes_after_fit:
            if attr not in ["fit_items", "fit_queries"]:
                client_params[attr] = getattr(model, attr)
        # client_params["_init_when_first_impl_arrived_args"] = model._init_when_first_impl_arrived_args
        client_params_path = path / "client_params.json"
        with client_params_path.open("w") as f:
            json.dump(client_params, f)
        model_type = model._get_implementation_type()
        model = model._impl

    init_args = model._init_args.copy()
    init_args["_model_name"] = str(model)
    init_args_path = path / "init_args.json"
    with init_args_path.open("w") as f:
        json.dump(init_args, f)

    df_dir = path / "dataframes"
    df_dir.mkdir(exist_ok=True)
    for name, df in model._dataframes.items():
        if df is not None:
            file_path = df_dir / f"{name}.parquet"
            if model_type == "pandas":
                df.to_parquet(str(file_path), index=False)
            elif model_type == "polars":
                df.write_parquet(str(file_path))

    if hasattr(model, "fit_queries") and model.fit_queries is not None:
        file_path = df_dir / "fit_queries.parquet"
        if model_type == "pandas":
            model.fit_queries.to_parquet(str(file_path), index=False)
        elif model_type == "polars":
            model.fit_queries.write_parquet(str(file_path))

    if hasattr(model, "fit_items") and model.fit_items is not None:
        file_path = df_dir / "fit_items.parquet"
        if model_type == "pandas":
            model.fit_items.to_parquet(str(file_path), index=False)
        elif model_type == "polars":
            model.fit_items.write_parquet(str(file_path))

    if hasattr(model, "study"):
        save_picklable_to_parquet_core(model.study, join(path, "study"))


def _save_spark(  # TODO: Think how to save model's implementations on pandas and polars without spark
    model: Union[BaseRecommenderClient, _BaseRecommenderSparkImpl], path: Union[str, Path], overwrite: bool = False
):
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
    if is_many_frameworks_model and model.is_fitted:
        client_params = {}
        client_params["model_class"] = str(model)
        client_params["impl_class"] = str(model._impl)
        impl_type = model._get_implementation_type()
        for attr in model.attributes_after_fit:
            if attr not in ["fit_items", "fit_queries"]:
                client_params[attr] = getattr(model, attr)
        # client_params["_init_when_first_impl_arrived_args"] = model._init_when_first_impl_arrived_args
        client_params = spark.read.json(spark.sparkContext.parallelize([json.dumps(client_params)]))
        client_params.coalesce(1).write.mode("overwrite").option("ignoreNullFields", "false").json(
            join(path, "spark_client_params.json")
        )
        model = model._impl
    fs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(join(path, "model")))
    model._save_model(join(path, "model"))
    init_args = model._init_args
    init_args["_model_name"] = str(model)
    dataframes = model._dataframes

    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    df.coalesce(1).write.mode("overwrite").option("ignoreNullFields", "false").json(join(path, "init_args.json"))
    # TODO: let's use repartition(1) instead of coalesce - its faster and more optimize

    df_path = join(path, "dataframes")
    fs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(join(path, "dataframes")))
    for name, df in dataframes.items():
        if df is not None:
            if isinstance(df, pd.DataFrame):
                df = convert2spark(df)
            elif isinstance(df, pl.DataFrame):
                df = convert2spark(df)
            df.write.mode("overwrite").parquet(join(df_path, name))

    if hasattr(model, "fit_queries"):
        if is_many_frameworks_model and impl_type in ["pandas", "polars"]:
            df = convert2spark(model.fit_queries)
            print(f'fit_queries: polars: {join(df_path, "fit_queries")=}')
        else:
            df = model.fit_queries
        df.write.mode("overwrite").parquet(join(df_path, "fit_queries"))
        print(f'fit_queries: spark: {join(df_path, "fit_queries")=}')
    if hasattr(model, "fit_items"):
        if is_many_frameworks_model and impl_type in ["pandas", "polars"]:
            df = convert2spark(model.fit_items)
            print(f'fit_items: polars: {join(df_path, "fit_items")=}')
        else:
            df = model.fit_items
        df.write.mode("overwrite").parquet(join(df_path, "fit_items"))
        print(f'fit_items: spark: {join(df_path, "fit_items")=}')
    if hasattr(model, "study"):
        save_picklable_to_parquet(model.study, join(path, "study"))


def load(path: Union[str, Path], model_type=None):
    """
    Load saved model from disk

    :param path: path to model folder
    :return: Restored trained model
    """
    path = Path(path)
    client_params_path = path / "spark_client_params.json"
    if not client_params_path.exists():
        model = _load_core(path, model_type)
    else:
        model = _load_spark(path, model_type)
    return model


def _load_core(path: str, model_type=None) -> Union[_BaseRecommenderSparkImpl, BaseRecommenderClient]:
    """
    Load saved model from disk

    :param path: path to model folder
    :return: Restored trained model
    """
    path = Path(path)
    # Load the initialization arguments from the JSON file
    init_args_path = path / "init_args.json"
    with init_args_path.open("r") as f:
        args = json.load(f)
    name = args["_model_name"]
    del args["_model_name"]
    model_class = model_type if model_type is not None else globals()[name]
    if name in [i.__name__ for i in implementations_list]:
        client_params_path = path / "client_params.json"
        with client_params_path.open("r") as f:
            client_params = json.load(f)
        model = globals()[client_params["model_class"]](**args)
        impl_class = globals()[client_params["impl_class"]](**args)
        model._impl = impl_class
        for attr in model.attributes_after_fit:
            if attr not in ["fit_items", "fit_queries"]:
                setattr(model._impl, attr, client_params[attr])
        # client_params["_init_when_first_impl_arrived_args"] = model._init_when_first_impl_arrived_args
    else:
        model = model_class(**args)

    df_dir = path / "dataframes"
    if df_dir.exists():
        for file in df_dir.glob("*.parquet"):
            attr_name = file.stem
            if model._get_implementation_type() == "pandas":
                df = pd.read_parquet(file)
            elif model._get_implementation_type() == "polars":
                df = pl.read_parquet(file)
            setattr(model, attr_name, df)

    model_model_path = path / "model"
    model._load_model(str(model_model_path))

    df_dir = path / "fit_queries"
    if df_dir.exists():
        df = pd.read_parquet(df_dir)
        setattr(model, "fit_queries", df)

    df_dir = path / "fit_items"
    if df_dir.exists():
        df = pd.read_parquet(df_dir)
        setattr(model, "fit_items", df)

    study_path = path / "study.pkl"
    if study_path.exists():
        with study_path.open("rb") as f:
            model.study = pickle.load(f)
    else:
        model.study = None

    return model


def _load_spark(path: str, model_type=None) -> _BaseRecommenderSparkImpl:
    """
    Load saved model from disk

    :param path: path to model folder
    :return: Restored trained model
    """
    spark = State().session  # TODO: Think how to load model's implementations on pandas and polars without spark
    args = spark.read.json(join(path, "init_args.json")).first().asDict(recursive=True)
    name = args["_model_name"]
    del args["_model_name"]
    model_class = model_type if model_type is not None else globals()[name]
    if name in [i.__name__ for i in implementations_list]:
        is_client = True
        client_params = spark.read.json(join(path, "spark_client_params.json")).first().asDict(recursive=True)
        model = globals()[client_params["model_class"]](**args)
        impl_class = globals()[client_params["impl_class"]](**args)
        realization = (
            "spark" if "Spark" in str(impl_class) else "pandas" if "Pandas" in str(impl_class) else
              "polars" if "Polars" in str(impl_class) else None
              )
        model._impl = impl_class
        model._assign_implementation_type(realization)
        for attr in model.attributes_after_fit:
            if attr not in ["fit_items", "fit_queries"]:
                setattr(model._impl, attr, client_params[attr])
        # client_params["_init_when_first_impl_arrived_args"] = model._init_when_first_impl_arrived_args
    else:
        is_client = False
        model = model_class(**args)

    dataframes_paths = get_list_of_paths(spark, join(path, "dataframes"))
    for dataframe_path in dataframes_paths:
        print(f"Пытаемся найти {dataframe_path=}")
        if not is_client or is_client and model._get_implementation_type() == "spark":
            df = spark.read.parquet(dataframe_path)
        elif is_client and model._get_implementation_type == "pandas":
            df = convert2pandas(spark.read.parquet(dataframe_path))
        elif is_client and model._get_implementation_type == "polars":
            df = convert2polars(spark.read.parquet(dataframe_path))
        else:
            df = None
        if df is not None:
            print("------------------------------------------============= ============= =========\n")
            print(dataframe_path)
            print(df)
            attr_name = dataframe_path.split("/")[-1]
            setattr(model, attr_name, df)
        else:
            print(f"Not found {is_client=} , {model._get_implementation_type()=},{dataframe_path=}")

    df_dir = path / "fit_queries"
    if df_dir.exists():
        if not is_client or is_client and model._get_implementation_type() == "spark":
            df = spark.read.parquet(df_dir)
        elif is_client and model._get_implementation_type == "pandas":
            df = convert2pandas(spark.read.parquet(df_dir))
        elif is_client and model._get_implementation_type == "polars":
            df = convert2polars(spark.read.parquet(df_dir))
        setattr(model, "fit_queries", df)

    df_dir = path / "fit_items"
    if df_dir.exists():
        if not is_client or is_client and model._get_implementation_type() == "spark":
            df = spark.read.parquet(df_dir)
        elif is_client and model._get_implementation_type == "pandas":
            df = convert2pandas(spark.read.parquet(df_dir))
        elif is_client and model._get_implementation_type == "polars":
            df = convert2polars(spark.read.parquet(df_dir))
        setattr(model, "fit_items", df)
    else:
        print("Не найдено! fit_items")

    model_save_path = path / "model"
    print(f"{model_save_path.parent.exists()=}")
    model._load_model(model_save_path)
    fs = get_fs(spark)
    model.study = (
        load_pickled_from_parquet(join(path, "study"))
        if fs.exists(spark._jvm.org.apache.hadoop.fs.Path(join(path, "study")))
        else None
    )
    return model
    '''
