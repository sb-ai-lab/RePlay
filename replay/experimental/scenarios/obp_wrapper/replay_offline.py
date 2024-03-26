import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
from obp.policy.base import BaseOfflinePolicyLearner
from optuna import create_study
from optuna.samplers import TPESampler
from pyspark.sql import DataFrame

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.experimental.scenarios.obp_wrapper.obp_optuna_objective import OBPObjective
from replay.experimental.scenarios.obp_wrapper.utils import split_bandit_feedback
from replay.models.base_rec import BaseRecommender
from replay.utils.spark_utils import convert2spark


def obp2df(action: np.ndarray, reward: np.ndarray, timestamp: np.ndarray) -> Optional[pd.DataFrame]:
    """
    Converts OBP log to the pandas DataFrame
    """

    n_interactions = len(action)

    df = pd.DataFrame(
        {
            "user_idx": np.arange(n_interactions),
            "item_idx": action,
            "rating": reward,
            "timestamp": timestamp,
        }
    )

    return df


def context2df(context: np.ndarray, idx_col: np.ndarray, idx_col_name: str) -> Optional[pd.DataFrame]:
    """
    Converts OBP log to the pandas DataFrame
    """

    df1 = pd.DataFrame({idx_col_name + "_idx": idx_col})
    cols = [str(i) + "_" + idx_col_name for i in range(context.shape[1])]
    df2 = pd.DataFrame(context, columns=cols)

    return df1.join(df2)


@dataclass
class OBPOfflinePolicyLearner(BaseOfflinePolicyLearner):
    """
    Off-policy learner which wraps OBP data representation into replay format.

    :param n_actions: Number of actions.

    :param len_list: Length of a list of actions in a recommendation/ranking inferface,
                     slate size. When Open Bandit Dataset is used, 3 should be set.

    :param replay_model: Any model from replay library with fit, predict functions.

    :param dataset: Dataset of interactions (user_id, item_id, rating).
                Constructing inside the fit method. Used for predict of replay_model.
    """

    replay_model: Optional[BaseRecommender] = None
    log: Optional[DataFrame] = None
    max_usr_id: int = 0
    item_features: DataFrame = None
    _study = None
    _logger: Optional[logging.Logger] = None
    _objective = OBPObjective

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.feature_schema = FeatureSchema(
            [
                FeatureInfo(
                    column="user_idx",
                    feature_type=FeatureType.CATEGORICAL,
                    feature_hint=FeatureHint.QUERY_ID,
                ),
                FeatureInfo(
                    column="item_idx",
                    feature_type=FeatureType.CATEGORICAL,
                    feature_hint=FeatureHint.ITEM_ID,
                ),
                FeatureInfo(
                    column="rating",
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.RATING,
                ),
                FeatureInfo(
                    column="timestamp",
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.TIMESTAMP,
                ),
            ]
        )

    @property
    def logger(self) -> logging.Logger:
        """
        :return: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    def fit(
        self,
        action: np.ndarray,
        reward: np.ndarray,
        timestamp: np.ndarray,
        context: np.ndarray = None,
        action_context: np.ndarray = None,
    ) -> None:
        """
        Fits an offline bandit policy on the given logged bandit data.
        This `fit` method wraps bandit data and calls `fit` method for the replay_model.

        :param action: Actions sampled by the logging/behavior policy
                       for each data in logged bandit data, i.e., :math:`a_i`.

        :param reward: Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        :param timestamp: Moment of time when user interacted with corresponding item.

        :param context: Context vectors observed for each data, i.e., :math:`x_i`.

        :param action_context: Context vectors observed for each action.
        """

        log = convert2spark(obp2df(action, reward, timestamp))
        self.log = log

        user_features = None
        self.max_usr_id = reward.shape[0]

        if context is not None:
            user_features = convert2spark(context2df(context, np.arange(context.shape[0]), "user"))

        if action_context is not None:
            self.item_features = convert2spark(context2df(action_context, np.arange(self.n_actions), "item"))

        dataset = Dataset(
            feature_schema=self.feature_schema,
            interactions=log,
            query_features=user_features,
            item_features=self.item_features,
        )
        self.replay_model._fit_wrap(dataset)

    def predict(self, n_rounds: int = 1, context: np.ndarray = None) -> np.ndarray:
        """Predict best actions for new data.
        Action set predicted by this `predict` method can contain duplicate items.
        If a non-repetitive action set is needed, please use the `sample_action` method.

        :context: Context vectors for new data.

        :return: Action choices made by a classifier, which can contain duplicate items.
            If a non-repetitive action set is needed, please use the `sample_action` method.
        """

        user_features = None
        if context is not None:
            user_features = convert2spark(
                context2df(context, np.arange(self.max_usr_id, self.max_usr_id + n_rounds), "user")
            )

        users = convert2spark(pd.DataFrame({"user_idx": np.arange(self.max_usr_id, self.max_usr_id + n_rounds)}))
        items = convert2spark(pd.DataFrame({"item_idx": np.arange(self.n_actions)}))

        self.max_usr_id += n_rounds

        dataset = Dataset(
            feature_schema=self.feature_schema,
            interactions=self.log,
            query_features=user_features,
            item_features=self.item_features,
            check_consistency=False,
        )

        action_dist = self.replay_model._predict_proba(dataset, self.len_list, users, items, filter_seen_items=False)

        return action_dist

    def optimize(
        self,
        bandit_feedback: Dict[str, np.ndarray],
        val_size: float = 0.3,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: str = "ipw",
        budget: int = 10,
        new_study: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Optimize model parameters using optuna.
        Optimization is carried out over the IPW/DR/DM scores(IPW by default).

        :param bandit_feedback: Bandit log data with fields
            ``[action, reward, context, action_context,
            n_rounds, n_actions, position, pscore]`` as in OpenBanditPipeline.

        :param val_size: Size of validation subset.

        :param param_borders: Dictionary of parameter names with pair of borders
                              for the parameters optimization algorithm.

        :param criterion: Score for optimization. Available are `ipw`, `dr` and `dm`.

        :param budget: Number of trials for the optimization algorithm.

        :param new_study: Flag to create new study or not for optuna.

        :return: Dictionary of parameter names with optimal value of corresponding parameter.
        """

        bandit_feedback_train, bandit_feedback_val = split_bandit_feedback(bandit_feedback, val_size)

        if self.replay_model._search_space is None:
            self.logger.warning("%s has no hyper parameters to optimize", str(self))
            return None

        if self._study is None or new_study:
            self._study = create_study(direction="maximize", sampler=TPESampler())

        search_space = self.replay_model._prepare_param_borders(param_borders)
        if self.replay_model._init_params_in_search_space(search_space) and not self.replay_model._params_tried():
            self._study.enqueue_trial(self.replay_model._init_args)

        objective = self._objective(
            search_space=search_space,
            bandit_feedback_train=bandit_feedback_train,
            bandit_feedback_val=bandit_feedback_val,
            learner=self,
            criterion=criterion,
            k=self.len_list,
        )

        self._study.optimize(objective, budget)
        best_params = self._study.best_params
        self.replay_model.set_params(**best_params)
        return best_params
