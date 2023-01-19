from pyspark.ml.base import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame


class TestTransformer(Transformer):
    test_df_param = Param(Params._dummy(), "test_df_param", "some text")

    def _transform(self, dataser):
        raise NotImplementedError()

    def setTestDf(self, value: DataFrame):
        self.set(self.test_df_param, value)

    def getTestDf(self) -> DataFrame:
        return self.getOrDefault(self.test_df_param)
