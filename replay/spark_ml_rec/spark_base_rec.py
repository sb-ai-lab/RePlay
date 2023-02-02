from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from pyspark.ml import Estimator, Model
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import MLWritable, MLWriter, MLReadable, MLReader
from pyspark.sql import DataFrame


ParamMap = Dict[str, Any]


RECOMMENDATIONS_PREDICT_MODE = "recommendations"
PAIRS_PREDICT_MODE = "pairs"


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
        "Defines how transform method should behave itself: "
        "mode 'recommendations' means to make recommendations for incoming unique users / items, "
        "while mode 'pairs' means to make relevance estimations for incoming user/item pairs",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super(SparkBaseRecModelParams, self).__init__()
        self._setDefault(
            numRecommendations=10,
            filterSeenItems=True,
            recsFilePath=None,
            predictMode='recommendations'
        )

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
    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.getPredictMode() == "recommendations":
            result = self._transform_recommendations(dataset)
        elif self.getPredictMode() == "pairs":
            result = self._transform_pairs(dataset)
        else:
            raise ValueError(f"Unsupported predict mode {self.getPredictMode()}. "
                             f"Only the following values are valid: "
                             f"{[RECOMMENDATIONS_PREDICT_MODE, PAIRS_PREDICT_MODE]}")

        return result

    @abstractmethod
    def _transform_recommendations(self, dataset: DataFrame) -> DataFrame:
        ...

    @abstractmethod
    def _transform_pairs(self, dataset: DataFrame) -> DataFrame:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class SparkBaseRec(Estimator, SparkBaseRecParams, ABC):
    ...
