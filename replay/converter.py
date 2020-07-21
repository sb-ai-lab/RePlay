"""
Модуль содержит функции, относящиеся к автоматической конвертации между форматами данных.
"""

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame

from replay.constants import AnyDataFrame
from replay.session_handler import State


def convert(
    data_frame: AnyDataFrame, to_type: type = SparkDataFrame
) -> AnyDataFrame:
    """
    Обеспечивает конвертацию данных в спарк и обратно.

    :param data_frame: данные в формате датафрейма пандас или спарк,
        либо объект датасета, в котором лежат датафреймы поддежриваемых форматов.
    :param to_type: в какой тип конвертировать ``{pyspark.sql.DataFrame, pandas.DataFrame}``.
    :return: преобразованные данные, если на вход был подан датафрейм.
    """
    if isinstance(data_frame, to_type):
        return data_frame
    if to_type == SparkDataFrame:
        spark = State().session
        return spark.createDataFrame(data_frame)  # type: ignore
    if to_type == PandasDataFrame:
        return data_frame.toPandas()  # type: ignore
    raise NotImplementedError(f"Неизвестный выходной тип: {type(data_frame)}")
