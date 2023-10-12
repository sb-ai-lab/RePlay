# pylint: skip-file
import datetime
import functools
import inspect
import logging
import os
import warnings
from math import floor
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import psutil

from .types import PYSPARK_AVAILABLE, MissingImportType, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import SparkSession
else:
    SparkSession = MissingImportType  # type: ignore

logger_level_dict = {
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARN,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "TRACE": logging.NOTSET,
}

TFunc = Callable[..., Any]


def get_local_time() -> str:
    """
    Get current time

    Returns:
        str: current time
    """
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")


def ensure_dir(dir_path: str) -> None:
    """
    Make sure the directory exists, if it does not exist, create it.

    Args:
        str: directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def init_logger(log_path: Optional[str] = None, level: Optional[str] = "INFO") -> logging.Logger:
    """
    Init logging for the library.

    Args:
        log_path (str, optional): path for logging in file.
            If ``None``, then write in ./log/amazmemllib_{current_time}.log.
            Default: ``None``.
        level (str, optional): level for logging, default: ``INFO``.
            May be [``FATAL``, ``ERROR``, ``WARN``, ``INFO``, ``DEBUG``, ``TRACE``].
            If ``None``, then logging level is determined by the environment variable ``LOG_LEVEL``.

    Returns:
        logging.Logger: logger.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO") if level is None else level
    log_level = logger_level_dict[level_name]

    if not log_path:
        logroot = "./log/"
        dir_name = os.path.dirname(logroot)
        ensure_dir(dir_name)
        logfilename = f"amazmemllib_{get_local_time()}.log"
        log_path = os.path.join(logroot, logfilename)

    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.WARN)

    logger = logging.getLogger("amazmemllib")
    formatter = logging.Formatter(
        fmt="{'eventTime':'%(asctime)s.%(msecs)03dZ',"
        "'serviceName':'%(name)s','msg':'%(message)s','level':'%(levelname)s'}",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    fh.setLevel(log_level)
    logger.addHandler(fh)

    return logger


def get_spark_session(
    core_count: Optional[int] = None,
    spark_memory: Optional[int] = None,
    shuffle_partitions: Optional[int] = None,
    enable_hive_support: bool = True,
) -> SparkSession:  # pragma: no cover
    """
    Get default SparkSession

    Args:
        core_count (int, optional): Count of cores to execute, ``-1`` means using all available cores.
            If ``None`` then checking out environment variable ``AMAZMEMLLIB_SPARK_CORE_COUNT``,
            if variable is not set then using ``-1``.
            Default: ``None``.
        spark_memory (int, optional): GB of memory allocated for Spark, ``None`` means using 70% of RAM.
            If ``None`` then checking out environment variable ``AMAZMEMLLIB_SPARK_MEMORY``,
            if variable is not set then using ``None``.
            Default: ``None``.
        shuffle_partitions (int, optional): Number of partitions,
            default: ``None`` (triple CPU count).
        enable_hive_support (bool, optional): Hive support,
            default: ``True``.

    Returns:
        SparkSession: Created or existed SparkSession.
    """
    if SparkSession.getActiveSession():
        session = SparkSession.getActiveSession()
        assert session is not None
        return session

    logger = logging.getLogger("amazmemllib")
    if core_count is None:  # checking out env variable
        core_count = int(os.environ.get("AMAZMEMLLIB_SPARK_CORE_COUNT", "-1"))
        logger.info(f"SparkSession using {core_count} cores.")

    if spark_memory is None:  # checking out env variable
        env_var = os.environ.get("AMAZMEMLLIB_SPARK_MEMORY")
        if env_var is not None:
            spark_memory = int(env_var)
            logger.info(f"SparkSession using {spark_memory}Gb of RAM.")

    if spark_memory is None:  # variable is not set in args and in env variable
        spark_memory = floor(psutil.virtual_memory().total / 1024**3 * 0.7)
    driver_memory = f"{spark_memory}g"

    if shuffle_partitions is None:
        cpu_count = os.cpu_count() or 1
        shuffle_partitions = cpu_count * 3

    spark_session_builder = (
        SparkSession.builder.config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .master(f"local[{'*' if core_count == -1 else core_count}]")
    )

    if enable_hive_support:
        spark_session_builder.enableHiveSupport()

    return spark_session_builder.getOrCreate()


def track_performance(func: TFunc) -> TFunc:
    def wrap(*args: list, **kwargs: dict) -> Any:
        logger = logging.getLogger("amazmemllib")
        start = perf_counter()
        func_result = func(*args, **kwargs)
        logger.debug(
            f"{args[0].__class__.__name__} :: {func.__name__} ::" f"{round((perf_counter() - start) * 1000, 2)}ms"
        )
        return func_result

    return wrap


def set_color(log: str, color: str, highlight: bool = True) -> str:
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except ValueError:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def check_type(f: TFunc, arguments: Dict[str, Any], args_names: Sequence[str]) -> None:
    type_hints = get_type_hints(f)
    for arg in args_names:
        check_type_obj = type_hints[arg]
        if get_origin(check_type_obj) is Union:
            check_type_obj = get_args(check_type_obj)
        if not isinstance(arguments[arg], check_type_obj):
            raise TypeError(f"\'{arg}\' did not pass type check. Given: {arguments[arg]}, should be {check_type_obj}")


def check_dataframe_type(*args_to_check: str) -> TFunc:
    def decorator_func(func: TFunc) -> TFunc:
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
            check_type(func, extended_kwargs, args_to_check)
            return func(*args, **kwargs)

        return wrap_func

    return decorator_func


class SparkCollectToMasterWarning(Warning):  # pragma: no cover
    pass


@check_dataframe_type("data")
def spark_to_pandas(data: SparkDataFrame, allow_collect_to_master: bool = False) -> PandasDataFrame:  # pragma: no cover
    """
    Convert Spark DataFrame to Pandas DataFrame.

    Args:
        data (PySpark DataFrame): Spark DataFrame.
        allow_collect_to_master (bool): Flag allowing spark to make a collection to the master node, default: ``False``.

    Returns:
        Pandas DataFrame: Converted dataframe.
    """
    if not allow_collect_to_master:
        warnings.warn(
            "Spark Data Frame is collected to master node, this may lead to OOM exception for larger dataset. "
            "To remove this warning set allow_collect_to_master=True in the recommender constructor.",
            SparkCollectToMasterWarning,
        )
    return data.toPandas()


@check_dataframe_type("data")
def pandas_to_spark(data: PandasDataFrame) -> SparkDataFrame:  # pragma: no cover
    """
    Convert Pandas DataFrame to Spark DataFrame.

    Args:
        data (Pandas DataFrame): Pandas DataFrame.

    Returns:
        Spark DataFrame: Converted dataframe.
    """
    spark = get_spark_session()

    # Warning should disappear once we migrate to new Spark version
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        spark_df = spark.createDataFrame(data)

    return spark_df
