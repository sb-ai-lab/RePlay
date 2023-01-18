from typing import overload, cast, Optional, Union, Iterable

from pyspark.ml import Model
from pyspark.ml._typing import ParamMap
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame

from replay.models.base_rec import UserRecommender
from replay.spark_ml_rec.spark_rec_base import SparkBaseRec, SparkBaseRecModel


class SparkUserRecModelParams(Params):
    transientUserFeatures = Param(
        Params._dummy(),
        "transientUserFeatures",
        "whatever or not to save the dataframe with user features",
        typeConverter=TypeConverters.toBoolean
    )

    def getTransientUserFeatures(self) -> bool:
        return self.getOrDefault(self.transientUserFeatures)

    def setTransientUserFeatures(self, value: bool):
        return self.set(self.transientUserFeatures, value)


class SparkUserRecParams(SparkUserRecModelParams):
    pass


class SparkUserRecModel(SparkBaseRecModel, SparkUserRecParams):
    # TODO: custom reader and writer required
    def __init__(self,
                 model: UserRecommender,
                 user_features: DataFrame,
                 transient_user_features: bool = False):
        super().__init__()
        self._model = model
        self._user_features = user_features
        self.setTransientUserFeatures(transient_user_features)

    def getUserFeatures(self) -> DataFrame:
        return self._user_features

    def setUserFeatures(self, value: DataFrame):
        self._user_features = value

    def _transform_recommendations(self, dataset: DataFrame, params: Optional[ParamMap]) -> DataFrame:
        return self._model.predict(
            user_features=self._user_features,
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
            user_features=self._user_features,
            recs_file_path=(
                params.get(self.recsFilePath, self.getRecsFilePath())
                if params else self.getRecsFilePath()
            )
        )

    def predict(
            self,
            user_features: DataFrame,
            k: int,
            log: Optional[DataFrame] = None,
            users: Optional[Union[DataFrame, Iterable]] = None,
            items: Optional[Union[DataFrame, Iterable]] = None,
            filter_seen_items: bool = True,
            recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        return self._model.predict(user_features, k, log, users, items, filter_seen_items, recs_file_path)

    def predict_pairs(
            self,
            pairs: DataFrame,
            user_features: DataFrame,
            log: Optional[DataFrame] = None,
            recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        return self._model.predict_pairs(pairs, user_features, log, recs_file_path)


class SparkUserRec(SparkBaseRec, SparkUserRecParams):
    def __init__(self, model: UserRecommender, user_features: DataFrame, transient_user_features: bool = False):
        super().__init__()
        self._model = model
        self._user_features = user_features
        self.setTransientUserFeatures(transient_user_features)

    def _do_fit(self, log: DataFrame, user_features: DataFrame):
        model = cast(UserRecommender, self._model.copy())
        model.fit(log, user_features=self._user_features)
        return SparkUserRecModel(
            model,
            self._user_features,
            self.getTransientUserFeatures()
        )

    def _fit(self, log: DataFrame):
        return self._do_fit(log, user_features=self._user_features)

    # TODO: fix this warning
    @overload
    def fit(
            self,
            log: DataFrame,
            user_features: DataFrame,
    ) -> Model:
        return self._do_fit(log, user_features)
