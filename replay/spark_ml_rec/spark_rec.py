from typing import Optional, Union, Iterable, cast, Tuple

from pyspark.ml._typing import ParamMap
from pyspark.ml.param import Param, Params
from pyspark.sql import DataFrame

from replay.models import Recommender
from replay.spark_ml_rec.spark_base_rec import SparkBaseRecModel, SparkBaseRec, SparkBaseRecModelParams


class SparkRecModelParams(SparkBaseRecModelParams):
    users = Param(Params._dummy(), "users", "a dataframe with users to make recommendations for")
    items = Param(Params._dummy(), "items", "a dataframe with items to make recommendations with")

    def getUsers(self) -> DataFrame:
        return self.getOrDefault(self.users)

    def setUsers(self, value: DataFrame):
        self.set(self.users, value)

    def getItems(self) -> DataFrame:
        return self.getOrDefault(self.items)

    def setItems(self, value: DataFrame):
        self.set(self.items, value)


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

    def _transform_recommendations(self, dataset: DataFrame, params: Optional[ParamMap]) -> DataFrame:
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

    def _transform_pairs(self, dataset: DataFrame, params: Optional[ParamMap]) -> DataFrame:
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


class SparkRec(SparkBaseRec):
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
