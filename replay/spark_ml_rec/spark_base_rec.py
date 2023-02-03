import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, cast

from pyspark.ml import Estimator, Model
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import MLWriter, DefaultParamsWritable, \
    DefaultParamsReader, DefaultParamsReadable
from pyspark.sql import DataFrame

from replay.model_handler import save, load
from replay.models.base_rec import BaseRecommender
from replay.session_handler import State
from replay.spark_ml_rec.writer_reader import DataframeAwareDefaultParamsWriter

ParamMap = Dict[str, Any]


RECOMMENDATIONS_PREDICT_MODE = "recommendations"
PAIRS_PREDICT_MODE = "pairs"


def _get_class_fullname_and_name(obj) -> Tuple[str, str]:
    clazz = obj.__class__
    module = clazz.__module__
    if module == 'builtins':
        return clazz.__qualname__
    return f"{module}.{clazz.__qualname__}", clazz.__qualname__


class SparkBaseRecModelWriter(DataframeAwareDefaultParamsWriter):
    def __init__(self, instance):
        super().__init__(instance)

    def saveImpl(self, path: str) -> None:
        super().saveImpl(path)

        spark = State().session

        model = cast('SparkBaseRecModel', self.instance).model
        clazz, model_name = _get_class_fullname_and_name(model)

        (
            spark
            .createDataFrame(data=[{"class_name": clazz, "model_name": model_name}])
            .write
            .mode('overwrite' if self.shouldOverwrite else 'error')
            .json(os.path.join(path, "model_class.json"))
        )

        save(model, os.path.join(path, f"model_{model_name}"), overwrite=self.shouldOverwrite)


class SparkBaseRecModelReader(DefaultParamsReader):
    def __init__(self, cls):
        super().__init__(cls)

    def load(self, path: str):
        wrapper = cast('SparkBaseRecModel', super().load(path))

        spark = State().session
        _, model_name = (
            spark
            .read
            .json(os.path.join(path, "model_class.json"))
            .select("class_name", "model_name")
            .first()
        )

        wrapper._model = load(os.path.join(path, f"model_{model_name}"))

        return wrapper


class SparkBaseRecModelWritable(DefaultParamsWritable):
    def write(self) -> MLWriter:
        return SparkBaseRecModelWriter(self)


class SparkBaseRecModelReadable(DefaultParamsReadable):
    @classmethod
    def read(cls) -> SparkBaseRecModelReader:
        return SparkBaseRecModelReader(cls)


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


class SparkBaseRecModel(Model, SparkBaseRecModelParams, SparkBaseRecModelReadable, SparkBaseRecModelWritable, ABC):
    _model: Optional[BaseRecommender]

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name or type(self._model).__name__

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
    def model(self) -> BaseRecommender:
        ...


class SparkBaseRec(Estimator, SparkBaseRecParams, ABC):
    ...


class SparkUserItemFeaturesModelParams(Params):
    transientUserFeatures = Param(
        Params._dummy(),
        "transientUserFeatures",
        "whatever or not to save the dataframe with user features",
        typeConverter=TypeConverters.toBoolean
    )

    userFeatures = Param(
        Params._dummy(),
        "userFeatures",
        "a dataframe containing user features"
    )

    transientItemFeatures = Param(
        Params._dummy(),
        "transientItemFeatures",
        "whatever or not to save the dataframe with user features",
        typeConverter=TypeConverters.toBoolean
    )

    itemFeatures = Param(
        Params._dummy(),
        "itemFeatures",
        "a dataframe containing user features"
    )

    def getTransientUserFeatures(self) -> bool:
        return self.getOrDefault(self.transientUserFeatures)

    def setTransientUserFeatures(self, value: bool):
        self.set(self.transientUserFeatures, value)

    def getTransientItemFeatures(self) -> bool:
        return self.getOrDefault(self.transientItemFeatures)

    def setTransientItemFeatures(self, value: bool):
        self.set(self.transientItemFeatures, value)

    def getUserFeatures(self) -> DataFrame:
        return self.getOrDefault(self.userFeatures)

    def setUserFeatures(self, value: DataFrame):
        self.set(self.userFeatures, value)

    def getItemFeatures(self) -> DataFrame:
        return self.getOrDefault(self.itemFeatures)

    def setItemFeatures(self, value: DataFrame):
        self.set(self.itemFeatures, value)
