"""
Using CQL implementation from `d3rlpy` package.
For 'alpha' version PySpark DataFrame are converted to Pandas
"""

from typing import Optional

import d3rlpy.algos.cql as CQL_d3rlpy
import numpy as np
import pandas as pd
from d3rlpy.argument_utility import (
    EncoderArg, QFuncArg, UseGPUArg, ScalerArg, ActionScalerArg,
    RewardScalerArg
)
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from pyspark.sql import DataFrame

from replay.data_preparator import DataPreparator
from replay.models import Recommender


class CQL(Recommender):
    r"""Conservative Q-Learning algorithm.

    CQL is a SAC-based data-driven deep reinforcement learning algorithm, which
    achieves state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::

        L(\theta_i) = \alpha\, \mathbb{E}_{s_t \sim D}
            \left[\log{\sum_a \exp{Q_{\theta_i}(s_t, a)}}
             - \mathbb{E}_{a \sim D} \big[Q_{\theta_i}(s_t, a)\big] - \tau\right]
            + L_\mathrm{SAC}(\theta_i)

    where :math:`\alpha` is an automatically adjustable value via Lagrangian
    dual gradient descent and :math:`\tau` is a threshold value.
    If the action-value difference is smaller than :math:`\tau`, the
    :math:`\alpha` will become smaller.
    Otherwise, the :math:`\alpha` will become larger to aggressively penalize
    action-values.

    In continuous control, :math:`\log{\sum_a \exp{Q(s, a)}}` is computed as
    follows.

    .. math::

        \log{\sum_a \exp{Q(s, a)}} \approx \log{\left(
            \frac{1}{2N} \sum_{a_i \sim \text{Unif}(a)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\text{Unif}(a)}\right]
            + \frac{1}{2N} \sum_{a_i \sim \pi_\phi(a|s)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\pi_\phi(a_i|s)}\right]\right)}

    where :math:`N` is the number of sampled actions.

    An implementation of this algorithm is heavily based on the corresponding implementation
    in the d3rlpy library (see https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/cql.py)

    The rest of optimization is exactly same as :class:`d3rlpy.algos.SAC`.

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float):
            learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for :math:`\alpha`.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\tau`.
        conservative_weight (float): constant weight to scale conservative loss.
        n_action_samples (int): the number of sampled actions to compute
            :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): flag to use SAC-style backup.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.cql_impl.CQLImpl): algorithm implementation.

    """

    k: int
    n_epochs: int
    model: CQL_d3rlpy.CQL

    _search_space = {
        "actor_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "critic_learning_rate": {"type": "loguniform", "args": [3e-5, 3e-4]},
        "n_epochs": {"type": "int", "args": [3, 20]},
        "temp_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "alpha_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "gamma": {"type": "loguniform", "args": [0.9, 0.999]},
        "n_critics": {"type": "int", "args": [2, 4]},
    }
    
    def __init__(
        self, *,
        k: int, n_epochs: int = 1,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 1e-4,
        alpha_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        alpha_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        initial_temperature: float = 1.0,
        initial_alpha: float = 1.0,
        alpha_threshold: float = 10.0,
        conservative_weight: float = 5.0,
        n_action_samples: int = 10,
        soft_q_backup: bool = False,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        **params
    ):
        super().__init__()
        self.k = k
        self.n_epochs = n_epochs
        self.model = CQL_d3rlpy.CQL(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            alpha_learning_rate=alpha_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            alpha_optim_factory=alpha_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            initial_alpha=initial_alpha,
            alpha_threshold=alpha_threshold,
            conservative_weight=conservative_weight,
            n_action_samples=n_action_samples,
            soft_q_backup=soft_q_backup,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            **params
        )

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        if user_features or item_features:
            message = f'CQL recommender does not support user/item features'
            self.logger.debug(message)

        users = users.toPandas().to_numpy().flatten()
        items = items.toPandas().to_numpy().flatten()

        user_item_pairs = pd.DataFrame({
            'user_idx': np.repeat(users, len(items)),
            'item_idx': np.tile(items, reps=len(users))
        })
        user_item_pairs['relevance'] = self.model.predict(user_item_pairs.to_numpy())

        # it doesn't explicitly filter seen items and doesn't return top k items
        # instead, it keeps all predictions as is to be filtered further by base methods
        return DataPreparator.read_as_spark_df(user_item_pairs)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        train: MDPDataset = self._prepare_data(log)
        self.model.fit(train, n_epochs=self.n_epochs)

    def _prepare_data(self, log: DataFrame) -> MDPDataset:
        # TODO: consider making calculations in Spark before converting to pandas
        user_logs = log.toPandas().sort_values('timestamp').groupby('user_idx')
        user_logs = pd.concat([
            user_logs.get_group(x) for x in user_logs.groups
        ])

        # reward top-K watched movies with 1, the others - with 0
        idxs = (
            user_logs
            .sort_values(['relevance', 'timestamp'], ascending=False)
            .groupby('user_idx')
            .head(self.k)
            .index
        )
        rewards = np.zeros(len(user_logs))
        rewards[idxs] = 1.0
        user_logs['rewards'] = rewards

        # every user has his own episode (the latest movie defined as terminal)
        user_change = user_logs.user_idx != user_logs.user_idx.shift()
        terminals = np.zeros(len(user_logs))
        terminals[user_change] = 1
        terminals[0] = 0
        user_logs['terminals'] = terminals

        train_dataset = MDPDataset(
            observations=np.array(user_logs[['user_idx', 'item_idx']]),
            actions=np.array(
                user_logs['relevance'] + 0.1 * np.random.randn(len(user_logs))
            )[:, None],
            rewards=user_logs['rewards'],
            terminals=user_logs['terminals']
        )
        return train_dataset
    
    @property
    def _init_args(self):
        args = dict(
            k=self.k,
            n_epochs=self.n_epochs,
        )
        args.update(**self.model.get_params())
        return args
