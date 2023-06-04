import logging
import pickle
from abc import abstractmethod
from typing import Dict, Optional, List

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from pyspark.ml._typing import T, M
from pyspark.ml.base import PredictionModel, Predictor, Model, Estimator
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import DataFrame, SparkSession

from replay.utils import (
    convert2spark,
    get_top_k_recs, )


class _ReRankerParams(HasLabelCol, HasPredictionCol, HasInputCols):
    pass


# class ReRanker(AbleToSaveAndLoad):
class ReRanker(Estimator, _ReRankerParams, DefaultParamsReadable, DefaultParamsWritable):
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
    ):
        super(ReRanker, self).__init__()
        self.setInputCols(input_cols or [])
        self.setLabelCol(label_col)
        self.setPredictionCol(prediction_col)

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    @abstractmethod
    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> 'ReRankerModel':
        """
        Fit the model which re-rank user-item pairs generated outside the models.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
            columns and features' columns
        :param fit_params: dict of parameters to pass to model.fit()
        """

    # @abstractmethod
    # def predict(self, data, k) -> DataFrame:
    #     """
    #     Re-rank data with the model and get top-k recommendations for each user.
    #
    #     :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
    #         columns and features' columns
    #     :param k: number of recommendations for each user
    #     """


class ReRankerModel(Model, _ReRankerParams, DefaultParamsReadable, DefaultParamsWritable):
    numRecommendations = Param(
        Params._dummy(), "numRecommendations", "Count of recommendations per user.", typeConverter=TypeConverters.toInt
    )

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

    def getNumRecommendations(self) -> int:
        return self.getOrDefault(self.numRecommendations)

    @abstractmethod
    def predict(self, value: DataFrame) -> DataFrame:
        """
        Predict label for the given features.
        """


class LamaWrap(ReRanker):
    """
    LightAutoML TabularPipeline binary classification model wrapper for recommendations re-ranking.
    Read more: https://github.com/sberbank-ai-lab/LightAutoML
    """

    @classmethod
    def load(cls, path: str, spark: Optional[SparkSession] = None):
        spark = spark or cls._get_spark_session()
        row = spark.read.parquet(path).first().asDict()
        model = pickle.loads(row["data"])
        wrap = LamaWrap()
        wrap.model = model
        return wrap

    def save(self, path: str, overwrite: bool = False, spark: Optional[SparkSession] = None):
        spark = spark or self._get_spark_session()
        data = pickle.dumps(self.model)

        spark.createDataFrame([{
            "classname": self.get_classname(),
            "data": data
        }]).write.parquet(path, mode='overwrite' if overwrite else 'error')

    def __init__(
        self,
        params: Optional[Dict] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize LightAutoML TabularPipeline with passed params/configuration file.

        :param params: dict of model parameters
        :param config_path: path to configuration file
        """
        self.model = TabularAutoML(
            task=Task("binary"),
            config_path=config_path,
            **(params if params is not None else {}),
        )

    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        """
        Fit the LightAutoML TabularPipeline model with binary classification task.
        Data should include negative and positive user-item pairs.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
            columns and features' columns. `Target` column should consist of zeros and ones
            as the model is a binary classification model.
        :param fit_params: dict of parameters to pass to model.fit()
            See LightAutoML TabularPipeline fit_predict parameters.
        """

        params = {"roles": {"target": "target"}, "verbose": 1}
        params.update({} if fit_params is None else fit_params)
        data = data.drop("user_idx", "item_idx")
        data_pd = data.toPandas()
        self.model.fit_predict(data_pd, **params)

    def predict(self, data: DataFrame, k: int) -> DataFrame:
        """
        Re-rank data with the model and get top-k recommendations for each user.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
            columns and features' columns
        :param k: number of recommendations for each user
        :return: spark dataframe with top-k recommendations for each user
            the dataframe columns are ``[user_idx, item_idx, relevance]``
        """
        data_pd = data.toPandas()
        candidates_ids = data_pd[["user_idx", "item_idx"]]
        data_pd.drop(columns=["user_idx", "item_idx"], inplace=True)
        self.logger.info("Starting re-ranking")
        candidates_pred = self.model.predict(data_pd)
        candidates_ids.loc[:, "relevance"] = candidates_pred.data[:, 0]
        self.logger.info(
            "%s candidates rated for %s users",
            candidates_ids.shape[0],
            candidates_ids["user_idx"].nunique(),
        )

        self.logger.info("top-k")
        return get_top_k_recs(
            recs=convert2spark(candidates_ids), k=k, id_type="idx"
        )
