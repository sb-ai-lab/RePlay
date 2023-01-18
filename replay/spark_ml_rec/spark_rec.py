from typing import Optional, Union, Iterable, cast, Tuple

from pyspark.ml._typing import ParamMap
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.sql import DataFrame

from replay.models import Recommender
from replay.spark_ml_rec.spark_rec_base import SparkBaseRecModel, SparkBaseRec


class SparkRecModelParams(Params):
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
        super(SparkRecModelParams, self).__init__()
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


class SparkRecParams(SparkRecModelParams):
    pass


class SparkRecModel(SparkBaseRecModel, SparkRecModelParams):
    def __init__(self,
                 model: Recommender,
                 num_recommendations: int = 10,
                 filter_seen_items: bool = True,
                 recs_file_path: Optional[str] = None,
                 predict_mode: str = "recommendations"):
        super().__init__()
        self._model = model
        self.setNumRecommendations(num_recommendations)
        self.setFilterSeenItems(filter_seen_items)

        if recs_file_path is not None:
            self.setRecsFilePath(recs_file_path)

        self.setPredictMode(predict_mode)

    def transform(self, dataset: DataFrame, params: Optional[ParamMap] = None) -> DataFrame:
        if self.predictMode == "recommendations":
            return self._model.predict(
                log=dataset,
                k=(
                    params.get(self.numRecommendations, self.getNumRecommendations())
                    if params else self.getNumRecommendations()
                ),
                filter_seen_items=(
                    params.get(self.filterSeenItems, self.getFilterSeenItems())
                    if params else self.getFilterSeenItems()
                ),
                recs_file_path=(
                    params.get(self.recsFilePath, self.getRecsFilePath())
                    if params else self.getRecsFilePath()
                )
            )
        else:
            return self._model.predict_pairs(
                pairs=dataset,
                recs_file_path=(
                    params.get(self.recsFilePath, self.getRecsFilePath())
                    if params else self.getRecsFilePath()
                )
            )

    def predict(
            self,
            log: DataFrame,
            k: int,
            users: Optional[Union[DataFrame, Iterable]] = None,
            items: Optional[Union[DataFrame, Iterable]] = None,
            filter_seen_items: bool = True,
            recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        return self._model.predict(log, k, users, items, filter_seen_items, recs_file_path)

    def predict_pairs(
            self,
            pairs: DataFrame,
            log: Optional[DataFrame] = None,
            recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        return self._model.predict_pairs(pairs, log, recs_file_path)

    def get_features(self, ids: DataFrame) -> Optional[Tuple[DataFrame, int]]:
        return self._model.get_features(ids)


class SparkRec(SparkBaseRec, SparkRecParams):
    def __init__(self, model: Recommender):
        super().__init__()
        self._model = model

    def _fit(self, log: DataFrame):
        model = cast(Recommender, self._model.copy())
        model.fit(log)
        return SparkRecModel(
            model,
            self.getNumRecommendations(),
            self.getFilterSeenItems(),
            self.getRecsFilePath(),
            self.getPredictMode()
        )
