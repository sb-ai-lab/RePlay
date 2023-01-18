from typing import Optional, overload, Union, List, Tuple, Iterable

from pyspark.ml import Estimator, Model
from pyspark.ml._typing import ParamMap, M
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import MLWritable, MLWriter, MLReadable, MLReader, R
from pyspark.sql import DataFrame

from replay.models import Recommender


class SparkRecommenderModelParams(Params):
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

    def __init__(self):
        super(SparkRecommenderModelParams, self).__init__()
        self._setDefault(numRecommendations=10, filterSeenItems=True)

    def getNumRecommendations(self) -> int:
        """
        Gets the value of numRecommendations or its default value.
        """
        return self.getOrDefault(self.numRecommendations)

    def getFilterSeen(self) -> bool:
        return self.getOrDefault(self.filterSeenItems)

    def getRecsFilePath(self) -> str:
        return self.getOrDefault(self.recsFilePath)

    def setNumRecommendations(self, value: int):
        self.set(self.numRecommendations, value)

    def setFilterSeenItems(self, value: bool):
        self.set(self.filterSeenItems, value)

    def setRecsFilePath(self, value: str):
        self.set(self.recsFilePath, value)


class SparkRecommenderParams(SparkRecommenderModelParams):
    pass


class SparkRecommenderModelWriter(MLWriter):
    def saveImpl(self, path: str) -> None:
        super().saveImpl(path)


class SparkRecommenderModelReader(MLReader):
    def load(self, path: str) -> R:
        return super().load(path)


class SparkRecommenderModelWritable(MLWritable):
    def write(self) -> MLWriter:
        return super().write()


class SparkRecommenderModelReadable(MLReadable):
    @classmethod
    def read(cls: Type[R]) -> MLReader[R]:
        return super().read()


class SparkRecommenderModel(Model,
                            SparkRecommenderModelParams,
                            SparkRecommenderModelReadable,
                            SparkRecommenderModelWritable):
    def __init__(self, model: Recommender):
        super().__init__()
        self._model = model

    def transform(self, log: DataFrame, params: Optional[ParamMap] = None) -> DataFrame:
        return self._model.predict(
            log=log,
            k=params.get(self.numRecommendations, self.getNumRecommendations()) if params else self.getNumRecommendations(),
            filter_seen_items=params.get(self.filterSeenItems, self.getFilterSeen()) if params else self.getFilterSeen(),
            recs_file_path=params.get(self.recsFilePath, self.getRecsFilePath()) if params else self.getRecsFilePath()
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


class SparkRecommender(Estimator, SparkRecommenderParams):
    def __init__(self, model: Recommender):
        super().__init__()
        self._model = model

    def _fit(self, log: DataFrame):
        model: Recommender = self._model.copy()
        model.fit(log)
        return SparkRecommenderModel(model)
