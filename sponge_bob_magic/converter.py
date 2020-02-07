from pyspark.sql import DataFrame as SparkDataFrame
from pandas import DataFrame as PandasDataFrame

from sponge_bob_magic.session_handler import State


class Converter:
    """
    Обеспечивает конвертацию данных в спарк и обратно.
    """
    def __init__(self, log):
        if isinstance(log, SparkDataFrame):
            self.type = 'spark'
        elif isinstance(log, PandasDataFrame):
            self.type = 'pandas'

    def __call__(self, log):
        if self.type == 'spark':
            return log
        elif self.type == 'pandas':
            spark = State().session
            if isinstance(log, PandasDataFrame):
                return spark.createDataFrame(log)
            else:
                return log.toPandas()
