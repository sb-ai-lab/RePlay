"""
Модуль содержит функции, относящиеся к автоматической конвертации между форматами данных.
"""
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame

from sponge_bob_magic.constants import AnyDataFrame
from sponge_bob_magic.session_handler import State


def convert(*args, to: type = SparkDataFrame) -> AnyDataFrame:
    """
    Обеспечивает конвертацию данных в спарк и обратно.

    :param args: данные в формате датафрейма пандас или спарк,
        либо объект датасета, в котором лежат датафреймы поддежриваемых форматов.
    :param to: в какой тип конвертироавть ``{pyspark.sql.DataFrame, pandas.DataFrame}``.
    :return: преобразованные данные, если на вход был подан датафрейм.
    """
    if len(args) > 1:
        return tuple([convert(arg, to=to) for arg in args])
    data_frame = args[0]
    if data_frame is None:
        return None
    if type(data_frame) == to:
        return data_frame
    if to == SparkDataFrame:
        spark = State().session
        return spark.createDataFrame(data_frame)
    if to == PandasDataFrame:
        return data_frame.toPandas()
    raise NotImplementedError(
        f"{type(data_frame)} conversion is not implemented"
    )
