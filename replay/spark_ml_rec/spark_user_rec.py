from typing import cast, Optional, Union, Iterable

from pyspark.sql import DataFrame

from replay.models.base_rec import UserRecommender, BaseRecommender
from replay.spark_ml_rec.spark_base_rec import SparkBaseRec, SparkBaseRecModel, SparkUserItemFeaturesModelParams


class SparkUserRecModel(SparkBaseRecModel, SparkUserItemFeaturesModelParams):
    # TODO: custom reader and writer required
    def __init__(self,
                 model: Optional[UserRecommender] = None,
                 user_features: Optional[DataFrame] = None,
                 transient_user_features: bool = False,
                 name: Optional[str] = None
                 ):
        super().__init__(name)
        self._model = model
        self.setUserFeatures(user_features)
        self.setTransientUserFeatures(transient_user_features)

    @property
    def model(self) -> BaseRecommender:
        return self._model

    def _transform_recommendations(self, dataset: DataFrame) -> DataFrame:
        return self._model.predict(
            user_features=self.getUserFeatures(),
            log=dataset,
            k=self.getNumRecommendations(),
            filter_seen_items=self.getFilterSeenItems(),
            recs_file_path=self.getRecsFilePath()
        )

    def _transform_pairs(self, dataset: DataFrame) -> DataFrame:
        return self._model.predict_pairs(
            pairs=dataset,
            user_features=self.getUserFeatures(),
            recs_file_path=self.getRecsFilePath()
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


class SparkUserRec(SparkBaseRec, SparkUserItemFeaturesModelParams):
    def __init__(self,
                 model: Optional[UserRecommender] = None,
                 user_features: Optional[DataFrame] = None,
                 transient_user_features: bool = False):
        super().__init__()
        self._model = model
        self.setUserFeatures(user_features)
        self.setTransientUserFeatures(transient_user_features)

    def _do_fit(self, log: DataFrame, user_features: DataFrame):
        model = cast(UserRecommender, self._model.copy())
        model.fit(log, user_features=user_features)
        return SparkUserRecModel(
            model,
            user_features,
            self.getTransientUserFeatures()
        )

    def _fit(self, log: DataFrame):
        if self.getUserFeatures() is None:
            raise ValueError("userFeatures must be set")
        return self._do_fit(log, user_features=self.getUserFeatures())
