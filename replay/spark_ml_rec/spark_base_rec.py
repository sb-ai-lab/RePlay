from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from pyspark.ml import Estimator, Model
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import MLWritable, MLWriter, MLReadable, MLReader
from pyspark.sql import DataFrame


ParamMap = Dict[str, Any]


class SparkRecModelWriter(MLWriter):
    def saveImpl(self, path: str) -> None:
        # TODO: save model
        # TODO: save parameters
        raise NotImplementedError()


class SparkRecModelReader(MLReader):
    def load(self, path: str):
        # TODO: load model
        # TODO: load parameters
        raise NotImplementedError()


class SparkRecModelWritable(MLWritable):
    def write(self) -> MLWriter:
        return SparkRecModelWriter()


class SparkRecModelReadable(MLReadable):
    @classmethod
    def read(cls) -> SparkRecModelReader:
        return SparkRecModelReader()


class SparkBaseRecModelParams(Params):
    numRecommendations = Param(
        Params._dummy(),
        "numRecommendations",
        "number recommendations per user to make",
        typeConverter=TypeConverters.toInt
    )

    filterSeenItems = Param(
        Params._dummy(),
        "filterSeenItems",
        "filter or not previously seen items if they appear in recommendations predicted by the model",
        typeConverter=TypeConverters.toBoolean
    )

    recsFilePath = Param(
        Params._dummy(),
        "recsFilePath",
        "a file path where to dump recommendations",
        typeConverter=TypeConverters.toString,
    )

    predictMode = Param(
        Params._dummy(),
        "predictMode",
        "defines to make recommendations for incoming unique users / items "
        "or to make relevance estimations for incoming user/item pairs",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super(SparkBaseRecModelParams, self).__init__()
        self._setDefault(numRecommendations=10, filterSeenItems=True)

    def getNumRecommendations(self) -> int:
        """
        Gets the value of numRecommendations or its default value.
        """
        return self.getOrDefault(self.numRecommendations)

    def getFilterSeenItems(self) -> bool:
        return self.getOrDefault(self.filterSeenItems)

    def getRecsFilePath(self) -> Optional[str]:
        return self.getOrDefault(self.recsFilePath)

    def getPredictMode(self) -> str:
        return self.getOrDefault(self.predictMode)

    def setNumRecommendations(self, value: int):
        self.set(self.numRecommendations, value)

    def setFilterSeenItems(self, value: bool):
        self.set(self.filterSeenItems, value)

    def setRecsFilePath(self, value: str):
        self.set(self.recsFilePath, value)

    def setPredictMode(self, value: str):
        assert value in ["recommendations", "pairs_relevance_estimating"]

        self.set(self.predictMode, value)


class SparkBaseRecParams(SparkBaseRecModelParams):
    pass


class SparkBaseRecModel(Model, SparkBaseRecModelParams, SparkRecModelReadable, SparkRecModelWritable, ABC):
    def transform(self, dataset: DataFrame, params: Optional[ParamMap] = None) -> DataFrame:
        if self.predictMode == "recommendations":
            result = self._transform_recommendations(dataset, params)
        else:
            result = self._transform_pairs(dataset, params)

        return result

    @abstractmethod
    def _transform_recommendations(self, dataset: DataFrame, params: Optional[ParamMap]) -> DataFrame:
        ...

    @abstractmethod
    def _transform_pairs(self, dataset: DataFrame, params: Optional[ParamMap]) -> DataFrame:
        ...


class SparkBaseRec(Estimator, SparkBaseRecParams, ABC):
    ...
