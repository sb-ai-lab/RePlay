"""
Модуль содержит функции, относящиеся к автоматической конвертации между форматами данных.
"""
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame

from sponge_bob_magic.constants import AnyDataFrame
from sponge_bob_magic.session_handler import State


def convert(*args, to_type: type = SparkDataFrame) -> AnyDataFrame:
    """
    Обеспечивает конвертацию данных в спарк и обратно.

    :param args: данные в формате датафрейма пандас или спарк,
        либо объект датасета, в котором лежат датафреймы поддежриваемых форматов.
    :param to_type: в какой тип конвертировать ``{pyspark.sql.DataFrame, pandas.DataFrame}``.
    :return: преобразованные данные, если на вход был подан датафрейм.
    """
    if len(args) > 1:
        return tuple([convert(arg, to_type=to_type) for arg in args])
    data_frame = args[0]
    if data_frame is None:
        return None
    if isinstance(data_frame, to_type):
        return data_frame
    if to_type == SparkDataFrame:
        spark = State().session
        return spark.createDataFrame(data_frame)
    if to_type == PandasDataFrame:
        return data_frame.toPandas()
    raise NotImplementedError(f"Неизвестный выходной тип: {type(data_frame)}")
