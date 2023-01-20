from typing import List

from pyspark.ml import Estimator, Transformer
from pyspark.sql.dataframe import DataFrame

from replay.experiment import Experiment
from replay.metrics import MAP, NDCG, HitRate
from replay.spark_ml_rec.spark_base_rec import SparkBaseRec, SparkBaseRecModelParams
from replay.spark_ml_rec.spark_rec import SparkRec, SparkRecModel
from replay.splitters import Splitter


class SparkTrainTestSplitter(Estimator, SparkBaseRecModelParams):
    def __init__(self, splitter: Splitter, models: List[SparkBaseRec]):
        super().__init__()
        self._splitter = splitter
        self._models = models

    def _fit(self, log: DataFrame) -> Transformer:
        train, test = self._splitter.split(log)

        # TODO: should be synced with K
        list_metrics = [5, 10, 25, 50, 100, 1000]
        # TODO: set study to the transformer as a dict?
        e = Experiment(
            test,
            {
                MAP(): list_metrics,
                NDCG(): list_metrics,
                HitRate(): list_metrics,
            },
        )

        self._rec_models = dict()
        for model in self._models:
            rec_model: SparkRecModel = model.fit(train)

            # users = test.select("user_idx").distinct(),
            # log = train,

            # single intreface with many parameters
            # the wrappers decide for themselves what to call
            recs = rec_model.transform(test, {
                SparkBaseRecModelParams.numRecommendations: self.getNumRecommendations(),
                SparkBaseRecModelParams.filterSeenItems: self.getFilterSeenItems()
            })

            e.add_result(rec_model.getName(), recs)
            self._rec_models[rec_model.getName()] = rec_model

            # TODO: optionally save models ?

        # TODO: find best model
        return self._rec_models[e.best_model()]

