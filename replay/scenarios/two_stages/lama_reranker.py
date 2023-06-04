from typing import Optional, Dict, List

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from pyspark.sql import DataFrame

from replay.scenarios.two_stages.reranker import ReRanker, AutoMLParams, ReRankerModel
from replay.utils import get_top_k_recs, convert2spark


class LamaWrap(ReRanker, AutoMLParams):
    """
    LightAutoML TabularPipeline binary classification model wrapper for recommendations re-ranking.
    Read more: https://github.com/sberbank-ai-lab/LightAutoML
    """
    def __init__(
            self,
            automl_params: Optional[Dict] = None,
            config_path: Optional[str] = None,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10
    ):
        """
        Initialize LightAutoML TabularPipeline with passed params/configuration file.

        :param params: dict of model parameters
        :param config_path: path to configuration file
        """
        super(LamaWrap, self).__init__(input_cols, label_col, prediction_col, num_recommendations)

        self.setAutoMLParams(automl_params)
        self.setConfigPath(config_path)

    def _fit(self, dataset: DataFrame):
        """
        Fit the LightAutoML TabularPipeline model with binary classification task.
        Data should include negative and positive user-item pairs.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
            columns and features' columns. `Target` column should consist of zeros and ones
            as the model is a binary classification model.
        :param fit_params: dict of parameters to pass to model.fit()
            See LightAutoML TabularPipeline fit_predict parameters.
        """
        params = {"roles": {"target": self.getLabelCol()}, "verbose": 1}
        params.update({} if self.getAutoMLParams() is None else self.getAutoMLParams())
        # data_pd = dataset.drop("user_idx", "item_idx").toPandas()
        data_pd = dataset.select(self.getLabelCol(), *self.getInputCols()).toPandas()

        model = TabularAutoML(
            task=Task("binary"),
            config_path=self.getConfigPath(),
            **params,
        )
        model.fit_predict(data_pd, **params)

        return LamaWrapModel(
            input_cols=self.getInputCols(),
            label_col=self.getLabelCol(),
            prediction_col=self.getPredictionCol(),
            num_recommendations=self.getNumRecommendations(),
            automl_model=model
        )


class LamaWrapModel(ReRankerModel):
    def __init__(
            self,
            input_cols: Optional[List[str]] = None,
            label_col: str = "target",
            prediction_col: str = "relevance",
            num_recommendations: int = 10,
            automl_model: Optional[TabularAutoML] = None
    ):
        super(LamaWrapModel, self).__init__(input_cols, label_col, prediction_col, num_recommendations)
        self._automl_model = automl_model

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Re-rank data with the model and get top-k recommendations for each user.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
            columns and features' columns
        :param k: number of recommendations for each user
        :return: spark dataframe with top-k recommendations for each user
            the dataframe columns are ``[user_idx, item_idx, relevance]``
        """
        data_pd = dataset.toPandas()
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
            recs=convert2spark(candidates_ids), k=self.getNumRecommendations()
        )
