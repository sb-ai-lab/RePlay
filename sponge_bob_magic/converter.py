"""
Модуль содержит функции, относящиеся к автоматической конвертации между форматами данных.
"""
from pyspark.sql import DataFrame as SparkDataFrame
from pandas import DataFrame as PandasDataFrame

from sponge_bob_magic.session_handler import get_session

SPARK = "spark"
PANDAS = "pandas"


def convert(df, type_out=SPARK):
    """
    Обеспечивает конвертацию данных в спарк и обратно.

    :param df: лог с данными в формате датафрейма пандас или спарк
    :param type_out: текстовая строка, во что конвертировать ``{"pandas", "spark"}``.
    :return: преобразованные данные
    """
    type_in = get_type(df)
    if type_in == type_out:
        return df
    elif type_out == SPARK:
        spark = get_session()
        return spark.createDataFrame(df)
    elif type_out == PANDAS:
        return df.toPandas()


def get_type(obj):
    """Текстовое описание типа объекта"""
    obj_type = type(obj)
    if obj_type is PandasDataFrame:
        res = PANDAS
    elif obj_type is SparkDataFrame:
        res = SPARK
    else:
        raise NotImplementedError(f"{obj_type} conversion is not implemented")
    return res
