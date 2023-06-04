import logging
import pickle
from abc import ABC
from typing import Dict, Optional, List, Any

from pyspark.ml.base import Model, Estimator
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable

from replay.scenarios.two_stages.utils import PickledAndDefaultParamsReadable, PickledAndDefaultParamsWritable


class _ReRankerParams(HasLabelCol, HasPredictionCol, HasInputCols):
    numRecommendations = Param(
        Params._dummy(), "numRecommendations", "Count of recommendations per user.", typeConverter=TypeConverters.toInt
    )

    def getNumRecommendations(self) -> int:
        return self.getOrDefault(self.numRecommendations)

    def setNumRecommendations(self, value: int):
        raise NotImplementedError()


class AutoMLParams(Params):
    automlParams = Param(
        Params._dummy(),
        "automlParams",
        "Dictionary with parameters of automl serialized in string",
        typeConverter=TypeConverters.toString
    )

    configPath = Param(
        Params._dummy(), "configPath", "Path to a configuration file for AutoML", typeConverter=TypeConverters.toString
    )

    def setAutoMLParams(self, value: Dict[str, Any]):
        pickled_automl_params = pickle.dumps(value)
        self.set(self.automlParams, pickled_automl_params)
        return self

    def getAutoMLParams(self) -> Dict[str, Any]:
        pickled_automl_params = self.getOrDefault(self.automlParams)
        return pickle.loads(pickled_automl_params)

    def setConfigPath(self, value: str):
        self.set(self.configPath, value)
        return self

    def getConfigPath(self) -> str:
        return self.getOrDefault(self.configPath)


class ReRanker(Estimator, _ReRankerParams, PickledAndDefaultParamsReadable, PickledAndDefaultParamsWritable, ABC):
    """
    Base class for models which re-rank recommendations produced by other models.
    May be used as a part of two-stages recommendation pipeline.
    """

    _logger: Optional[logging.Logger] = None

    def __init__(
            self,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10
    ):
        super(ReRanker, self).__init__()
        self.setInputCols(input_cols or [])
        self.setLabelCol(label_col)
        self.setPredictionCol(prediction_col)
        self.setNumRecommendations(num_recommendations)

    def setNumRecommendations(self, value: int):
        self.set(self.numRecommendations, value)
        return self

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger


class ReRankerModel(Model, _ReRankerParams, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(
            self,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10
    ):
        super(ReRankerModel, self).__init__()
        self.setInputCols(input_cols or [])
        self.setLabelCol(label_col)
        self.setPredictionCol(prediction_col)
        self.setNumRecommendations(num_recommendations)

    def setInputCols(self, value: List[str]):
        self.set(self.inputCols, value)
        return self

    def setLabelCol(self, value: str):
        self.set(self.labelCol, value)
        return self

    def setPredictionCol(self, value: str):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self.set(self.predictionCol, value)
        return self

    def setNumRecommendations(self, value: int):
        self.set(self.numRecommendations, value)
        return self

    def predict(self, value):
        """
        Predict label for the given features.
        """
        return self.transform(value)
