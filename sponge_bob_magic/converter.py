"""
Модуль содержит функции, относящиеся к автоматической конвертации между форматами данных.
"""
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame

from sponge_bob_magic.session_handler import State

SPARK = "spark"
PANDAS = "pandas"


def convert(data_frame, type_out=SPARK):
    """
    Обеспечивает конвертацию данных в спарк и обратно.

    :param data_frame: лог с данными в формате датафрейма пандас или спарк
    :param type_out: текстовая строка, во что конвертировать ``{"pandas", "spark"}``.
    :return: преобразованные данные
    """
    type_in = get_type(data_frame)
    if type_in == type_out:
        return data_frame
    if type_out == SPARK:
        spark = State().session
        return spark.createDataFrame(data_frame)
    if type_out == PANDAS:
        return data_frame.toPandas()


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
