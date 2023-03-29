from pyspark.ml import Estimator
from pyspark.ml.base import Transformer
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.sql import DataFrame

from replay.data_preparator import Indexer


class SparkIndexerParams(Params):
    userCol = Param(Params._dummy(), "userCol", "a column that stores user id", TypeConverters.toString)
    itemCol = Param(Params._dummy(), "itemCol", "a column that stores item id", TypeConverters.toString)

    def getUserCol(self) -> str:
        return self.getOrDefault(self.userCol)

    def setUserCol(self, value: str):
        self.set(self.userCol, value)

    def getItemCol(self) -> str:
        return self.getOrDefault(self.itemCol)

    def setItemCol(self, value: str):
        self.set(self.itemCol, value)


class SparkIndexer(Estimator, SparkIndexerParams):
    def __init__(self, indexer: Indexer):
        super().__init__()
        self._indexer = indexer

    def _fit(self, log: DataFrame) -> Transformer:
        self._indexer.fit(users=log.select("user_idx"), items=log.select("item_idx"))
        return SparkIndexerTransformer(self._indexer)


class SparkIndexerTransformer(Transformer, SparkIndexerParams):
    def __init__(self, indexer: Indexer):
        super().__init__()
        self._indexer = indexer

    def _transform(self, dataset: DataFrame) -> DataFrame:
        df = self._indexer.transform(dataset)
        assert df is not None
        return df
