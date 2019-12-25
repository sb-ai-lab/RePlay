import collections
import logging
from typing import Dict, Any, List, Optional

from optuna import Study, Trial

from sponge_bob_magic.metrics.base_metrics import Metric
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.scenarios.base_objective import Objective
from sponge_bob_magic.scenarios.base_scenario import Scenario

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
            filter_seen_items=Scenario.filter_seen_items
        )
        logging.debug(f"-- Длина рекомендаций: {recs.count()}")

        logging.debug("-- Подсчет метрики в оптимизации")
        criterion_value = self._calculate_metrics(trial, recs,
                                                  self.split_data.test,
                                                  self.criterion, self.metrics,
                                                  self.k)

        return criterion_value
