from pyspark.sql import DataFrame as SparkDataFrame
from pandas import DataFrame as PandasDataFrame

from sponge_bob_magic.session_handler import State

def convert(df, type_out='spark'):
    """
    Обеспечивает конвертацию данных в спарк и обратно.
    """
    type_in = type(df)
    if type_in == type_out:
        return df
    elif type_out == 'spark':
        spark = State().session
        return spark.createDataFrame(df)
    elif type_out == 'pandas':
        return df.toPandas()

def get_type(obj):
    obj_type = type(obj)
    if obj_type is PandasDataFrame:
        res = 'pandas'
    elif obj_type is SparkDataFrame:
        res = 'spark'
    else:
        raise NotImplementedError(f'{obj_type} conversion is not implemented')
    return res


