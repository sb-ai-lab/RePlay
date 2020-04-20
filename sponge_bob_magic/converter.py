"""
Модуль содержит функции, относящиеся к автоматической конвертации между форматами данных.
"""
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame

from sponge_bob_magic.session_handler import State

SPARK = "spark"
PANDAS = "pandas"
supported_types = [SPARK, PANDAS]


def convert(df, type_out=SPARK):
    """
    Обеспечивает конвертацию данных в спарк и обратно.
    Так же умеет конвертировать датасеты.

    :param df: лог с данными в формате датафрейма пандас или спарк,
        либо объект датасета, в котором лежат датафреймы поддежриваемых форматов.
    :param type_out: текстовая строка, во что конвертировать ``{"pandas", "spark"}``.
    :return: преобразованные данные, если на вход был подан датафрейм.
        Датасеты преобразуются на месте.
    """
    if df is None:
        return None
    type_in = get_type(df)
    if type_in == type_out:
        return df
    elif type_out == SPARK:
        spark = State().session
        return spark.createDataFrame(df)
    elif type_out == PANDAS:
        return df.toPandas()


def get_type(obj, except_unknown=True):
    """Текстовое описание типа объекта"""
    if isinstance(obj, PandasDataFrame):
        res = PANDAS
    elif isinstance(obj, SparkDataFrame):
        res = SPARK
    elif except_unknown:
        raise NotImplementedError(f"{type(obj)} conversion is not implemented")
    else:
        res = type(obj)
    return res
