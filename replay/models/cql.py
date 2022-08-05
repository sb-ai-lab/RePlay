"""
Using CQL implementation from d3rlpy
For 'alpha' version PySpark DataFrame converts to Pandas
"""

from typing import Any, Dict, Optional

import d3rlpy.algos.cql as CQL_d3rlpy
import numpy as np
import pandas as pd
import pyspark.sql
import torch
import torch.nn as nn
from d3rlpy.dataset import MDPDataset
from pandas import DataFrame

from replay.data_preparator import DataPreparator
from replay.models.base_torch_rec import TorchRecommender


class CQL(TorchRecommender):
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
    
    def __init__(
        self, *,
        k: int, n_epochs: int = 1,
        **params
    ):
        super().__init__()
        self.k = k
        self.n_epochs = n_epochs
        self.model = CQL_d3rlpy.CQL(**params)

    def _predict(
        self,
        log: pyspark.sql.DataFrame,
        k: int,
        users: pyspark.sql.DataFrame,
        items: pyspark.sql.DataFrame,
        user_features: Optional[pyspark.sql.DataFrame] = None,
        item_features: Optional[pyspark.sql.DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> pyspark.sql.DataFrame:

        users = users.toPandas().to_numpy().flatten()
        items = items.toPandas().to_numpy().flatten()

        pred = pd.DataFrame()
        
        # progress = tqdm.auto.tqdm(users, desc='User')
        for user in users:
            matrix = pd.DataFrame({
                'user_idx': np.repeat(user, len(items)),
                'item_idx': items
            })
            matrix['relevance'] = self.model.predict(matrix.to_numpy())
            top = matrix.sort_values('relevance', ascending=False).head(k)
            pred = pd.concat((pred, top))

        return DataPreparator.read_as_spark_df(pred)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        
        train = self._prepare_data(log.toPandas(), self.k)
        self.model.fit(train, n_epochs=self.n_epochs)

    def _prepare_data(self, df, k: int) -> MDPDataset:
        gb = df.sort_values('timestamp').groupby('user_idx')    
        list_dfs = [gb.get_group(x) for x in gb.groups]
        df_s = pd.concat(list_dfs)

        #   reward top-K and later watched movies with 1, other - 0
        idxs = df_s.sort_values(['relevance', 'timestamp'], ascending=False).groupby('user_idx').head(k).index
        rewards = np.zeros(len(df_s))
        rewards[idxs] = 1
        df_s['rewards'] = rewards

        #   every user has his own episode (for the latest movie terminals has 1, other - 0)
        user_change = (df_s.user_idx != df_s.user_idx.shift())
        terminals = np.zeros(len(df_s))
        terminals[user_change] = 1
        terminals[0] = 0
        df_s['terminals'] = terminals

        train_dataset = MDPDataset(
            observations=np.array(df_s[['user_idx', 'item_idx']]),
            actions=np.array(
                df_s['relevance'] + 0.1 * np.random.randn(len(df_s))
            )[:, None],
            rewards=df_s['rewards'],
            terminals=df_s['terminals']
        )
        return train_dataset
    
    @property
    def _init_args(self):
        pass

    def _batch_pass(self, batch, model) -> Dict[str, Any]:
        pass

    def _loss(self, **kwargs) -> torch.Tensor:
        pass

    @staticmethod
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def _predict_by_user_pairs(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        item_count: int,
    ):
        pass
