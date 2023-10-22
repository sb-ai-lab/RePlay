import logging
from abc import abstractmethod
from typing import Dict, Optional

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from pyspark.sql import DataFrame

from replay.utils.spark_utils import (
    convert2spark,
    get_top_k_recs,
)


class ReRanker:
    """
    Base class for models which re-rank recommendations produced by other models.
    May be used as a part of two-stages recommendation pipeline.
    """

    _logger: Optional[logging.Logger] = None

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    @abstractmethod
    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        """
        Fit the model which re-rank user-item pairs generated outside the models.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
            columns and features' columns
        :param fit_params: dict of parameters to pass to model.fit()
        """

    @abstractmethod
    def predict(self, data, k) -> DataFrame:
        """
        Re-rank data with the model and get top-k recommendations for each user.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
            columns and features' columns
        :param k: number of recommendations for each user
        """


class LamaWrap(ReRanker):
    """
    LightAutoML TabularPipeline binary classification model wrapper for recommendations re-ranking.
    Read more: https://github.com/sberbank-ai-lab/LightAutoML
    """

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
