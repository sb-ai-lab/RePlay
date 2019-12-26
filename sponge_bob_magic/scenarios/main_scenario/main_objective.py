import collections
import logging
from typing import Dict, Any, List, Optional

from optuna import Study, Trial
from pyspark.sql import DataFrame

from sponge_bob_magic.metrics.base_metrics import Metric
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.scenarios.base_objective import Objective

SplitData = collections.namedtuple(
    "SplitData",
    "train predict_input test users items user_features item_features"
)


class MainObjective(Objective):
    def __init__(
            self,
            params_grid: Dict[str, Dict[str, Any]],
            study: Study,
            split_data: SplitData,
            recommender: Recommender,
            criterion: Metric,
            metrics: List[Metric],
            k: int = 10,
            context: Optional[str] = None,
            fallback_recs: Optional[DataFrame] = None,
            filter_seen_items: bool = False,
            path: str = None
    ):
        self.path = path
        self.metrics = metrics
        self.criterion = criterion
        self.context = context
        self.k = k
        self.split_data = split_data
        self.recommender = recommender
        self.study = study
        self.params_grid = params_grid
        self.filter_seen_items = filter_seen_items

        self.max_in_fallback_recs = (
            fallback_recs
            .agg({"relevance": "max"})
            .collect()[0][0]
        ) if fallback_recs is not None else 0

        self.fallback_recs = (
            fallback_recs
            .withColumnRenamed("context", "context_fallback")
            .withColumnRenamed("relevance", "relevance_fallback")
        ) if fallback_recs is not None else None

    def __call__(
            self,
            trial: Trial,
    ) -> float:
        params = self._suggest_all_params(trial, self.params_grid)
        self.recommender.set_params(**params)

        self._check_trial_on_duplicates(trial)
        self._save_study(self.study, self.path)

        logging.debug("-- Второй фит модели в оптимизации")
        self.recommender._fit_partial(self.split_data.train,
                                      self.split_data.user_features,
                                      self.split_data.item_features,
                                      path=None)

        logging.debug("-- Предикт модели в оптимизации")
        recs = self.recommender.predict(
            k=self.k,
            users=self.split_data.users, items=self.split_data.items,
            user_features=self.split_data.user_features,
            item_features=self.split_data.item_features,
            context=self.context,
            log=self.split_data.predict_input,
            filter_seen_items=self.filter_seen_items
        )

        logging.debug("-- Дополняем рекомендации fallback рекомендациями")
        recs = self._join_fallback_recs(recs, self.fallback_recs, self.k,
                                        self.max_in_fallback_recs)

        logging.debug("-- Подсчет метрики в оптимизации")
        criterion_value = self._calculate_metrics(trial, recs,
                                                  self.split_data.test,
                                                  self.criterion, self.metrics,
                                                  self.k)

        return criterion_value
