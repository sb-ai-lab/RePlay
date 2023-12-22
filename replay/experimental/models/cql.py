"""
Using CQL implementation from `d3rlpy` package.
"""
import io
import logging
import tempfile
import timeit
from typing import Any, Dict, Optional, Union

from d3rlpy.algos import CQLConfig, CQL as CQL_d3rlpy
import numpy as np
import torch
from d3rlpy.base import LearnableConfigWithShape
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import EncoderFactory, DefaultEncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from d3rlpy.models.q_functions import QFunctionFactory, MeanQFunctionFactory
from d3rlpy.preprocessing import (
    ObservationScaler,
    ActionScaler,
    RewardScaler,
)

from replay.data import get_schema
from replay.experimental.models.base_rec import Recommender
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame
from replay.utils.spark_utils import assert_omp_single_thread

if PYSPARK_AVAILABLE:
    from pyspark.sql import Window
    from pyspark.sql import functions as sf

timer = timeit.default_timer


class CQL(Recommender):
    """Conservative Q-Learning algorithm.

    CQL is a SAC-based data-driven deep reinforcement learning algorithm, which
    achieves state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::
        L(\\theta_i) = \\alpha\\, \\mathbb{E}_{s_t \\sim D}
        \\left[\\log{\\sum_a \\exp{Q_{\\theta_i}(s_t, a)}} - \\mathbb{E}_{a \\sim D} \\big[Q_{\\theta_i}(s_t, a)\\big] - \\tau\\right]
        + L_\\mathrm{SAC}(\\theta_i)

    where :math:`\alpha` is an automatically adjustable value via Lagrangian
    dual gradient descent and :math:`\tau` is a threshold value.
    If the action-value difference is smaller than :math:`\tau`, the
    :math:`\alpha` will become smaller.
    Otherwise, the :math:`\alpha` will become larger to aggressively penalize
    action-values.

    In continuous control, :math:`\log{\sum_a \exp{Q(s, a)}}` is computed as
    follows.

    .. math::
        \\log{\\sum_a \\exp{Q(s, a)}} \\approx \\log{\\left(
        \\frac{1}{2N} \\sum_{a_i \\sim \\text{Unif}(a)}^N
            \\left[\\frac{\\exp{Q(s, a_i)}}{\\text{Unif}(a)}\\right]
        + \\frac{1}{2N} \\sum_{a_i \\sim \\pi_\\phi(a|s)}^N
            \\left[\\frac{\\exp{Q(s, a_i)}}{\\pi_\\phi(a_i|s)}\\right]\\right)}

    where :math:`N` is the number of sampled actions.

    An implementation of this algorithm is heavily based on the corresponding implementation
    in the d3rlpy library (see https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/cql.py)

    The rest of optimization is exactly same as :class:`d3rlpy.algos.SAC`.

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        mdp_dataset_builder (MdpDatasetBuilder): the MDP dataset builder from users' log.
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for the actor.
        The available options are `[SGD, Adam or RMSprop]`.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for the critic.
        The available options are `[SGD, Adam or RMSprop]`.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for the temperature.
        The available options are `[SGD, Adam or RMSprop]`.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for :math:`\alpha`.
        The available options are `[SGD, Adam or RMSprop]`.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
        encoder factory for the actor.
        The available options are `['pixel', 'dense', 'vector', 'default']`.
        See d3rlpy.models.encoders.EncoderFactory for details.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
        encoder factory for the critic.
        The available options are `['pixel', 'dense', 'vector', 'default']`.
        See d3rlpy.models.encoders.EncoderFactory for details.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
        Q function factory. The available options are `['mean', 'qr', 'iqn', 'fqf']`.
        See d3rlpy.models.q_functions.QFunctionFactory for details.
        batch_size (int): mini-batch size.
        n_steps (int): Number of training steps.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficient.
        n_critics (int): the number of Q functions for ensemble.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\tau`.
        conservative_weight (float): constant weight to scale conservative loss.
        n_action_samples (int): the number of sampled actions to compute
        :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): flag to use SAC-style backup.
        use_gpu (Union[int, str, bool]): device option.
        If the value is boolean and True, cuda:0 will be used.
        If the value is integer, cuda:<device> will be used.
        If the value is string in torch device style, the specified device will be used.
        observation_scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
        The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
        action preprocessor. The available options are `['min_max']`.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
        reward preprocessor. The available options are
        `['clip', 'min_max', 'standard']`.
        impl (d3rlpy.algos.torch.cql_impl.CQLImpl): algorithm implementation.
    """

    mdp_dataset_builder: 'MdpDatasetBuilder'
    model: CQL_d3rlpy

    can_predict_cold_users = True

    _observation_shape = (2, )
    _action_size = 1

    _search_space = {
        "actor_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "critic_learning_rate": {"type": "loguniform", "args": [3e-5, 3e-4]},
        "temp_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "alpha_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "gamma": {"type": "loguniform", "args": [0.9, 0.999]},
        "n_critics": {"type": "int", "args": [2, 4]},
    }

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
            self,
            mdp_dataset_builder: 'MdpDatasetBuilder',

            # CQL inner params
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 3e-4,
            temp_learning_rate: float = 1e-4,
            alpha_learning_rate: float = 1e-4,
            actor_optim_factory: OptimizerFactory = AdamFactory(),
            critic_optim_factory: OptimizerFactory = AdamFactory(),
            temp_optim_factory: OptimizerFactory = AdamFactory(),
            alpha_optim_factory: OptimizerFactory = AdamFactory(),
            actor_encoder_factory: EncoderFactory = DefaultEncoderFactory(),
            critic_encoder_factory: EncoderFactory = DefaultEncoderFactory(),
            q_func_factory: QFunctionFactory = MeanQFunctionFactory(),
            batch_size: int = 64,
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
            use_gpu: Union[int, str, bool] = False,
            observation_scaler: ObservationScaler = None,
            action_scaler: ActionScaler = None,
            reward_scaler: RewardScaler = None,
            **params
    ):
        super().__init__()
        assert_omp_single_thread()

        if isinstance(actor_optim_factory, dict):
            local = {}
            local["config"] = {}
            local["config"]["params"] = dict(locals().items())
            local["config"]["type"] = "cql"
            local["observation_shape"] = self._observation_shape
            local["action_size"] = self._action_size
            deserialized_config = LearnableConfigWithShape.deserialize_from_dict(local)

            self.logger.info('-- Desiarializing CQL parameters')
            actor_optim_factory = deserialized_config.config.actor_optim_factory
            critic_optim_factory = deserialized_config.config.critic_optim_factory
            temp_optim_factory = deserialized_config.config.temp_optim_factory
            alpha_optim_factory = deserialized_config.config.alpha_optim_factory
            actor_encoder_factory = deserialized_config.config.actor_encoder_factory
            critic_encoder_factory = deserialized_config.config.critic_encoder_factory
            q_func_factory = deserialized_config.config.q_func_factory
            observation_scaler = deserialized_config.config.observation_scaler
            action_scaler = deserialized_config.config.action_scaler
            reward_scaler = deserialized_config.config.reward_scaler
            # non-model params
            mdp_dataset_builder = MdpDatasetBuilder(**mdp_dataset_builder)

        self.mdp_dataset_builder = mdp_dataset_builder
        self.n_steps = n_steps

        self.model = CQLConfig(
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
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            initial_alpha=initial_alpha,
            alpha_threshold=alpha_threshold,
            conservative_weight=conservative_weight,
            n_action_samples=n_action_samples,
            soft_q_backup=soft_q_backup,
            observation_scaler=observation_scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            **params
        ).create(device=use_gpu)

        # explicitly create the model's algorithm implementation at init stage
        # despite the lazy on-fit init convention in d3rlpy a) to avoid serialization
        # complications and b) to make model ready for prediction even before fitting
        self.model.create_impl(
            observation_shape=self._observation_shape,
            action_size=self._action_size
        )

    def _fit(
        self,
        log: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> None:
        mdp_dataset: MDPDataset = self.mdp_dataset_builder.build(log)
        self.model.fit(mdp_dataset, self.n_steps)

    @staticmethod
    def _predict_pairs_inner(
        model: bytes,
        user_idx: int,
        items: np.ndarray,
    ) -> PandasDataFrame:
        user_item_pairs = PandasDataFrame({
            'user_idx': np.repeat(user_idx, len(items)),
            'item_idx': items
        })

        # deserialize model policy and predict items relevance for the user
        policy = CQL._deserialize_policy(model)
        items_batch = user_item_pairs.to_numpy()
        user_item_pairs['relevance'] = CQL._predict_relevance_with_policy(policy, items_batch)

        return user_item_pairs

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        available_items = items.toPandas()["item_idx"].values
        policy_bytes = self._serialize_policy()

        def grouped_map(log_slice: PandasDataFrame) -> PandasDataFrame:
            return CQL._predict_pairs_inner(
                model=policy_bytes,
                user_idx=log_slice["user_idx"][0],
                items=available_items,
            )[["user_idx", "item_idx", "relevance"]]

        # predict relevance for all available items and return them as is;
        # `filter_seen_items` and top `k` params are ignored
        self.logger.debug("Predict started")
        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        return users.groupby("user_idx").applyInPandas(grouped_map, rec_schema)

    def _predict_pairs(
        self,
        pairs: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> SparkDataFrame:
        policy_bytes = self._serialize_policy()

        def grouped_map(user_log: PandasDataFrame) -> PandasDataFrame:
            return CQL._predict_pairs_inner(
                model=policy_bytes,
                user_idx=user_log["user_idx"][0],
                items=np.array(user_log["item_idx_to_pred"][0]),
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Calculate relevance for user-item pairs")
        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        return (
            pairs
            .groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx_to_pred"))
            .join(log.select("user_idx").distinct(), on="user_idx", how="inner")
            .groupby("user_idx")
            .applyInPandas(grouped_map, rec_schema)
        )

    @property
    def _init_args(self) -> Dict[str, Any]:
        return {
            # non-model hyperparams
            "mdp_dataset_builder": self.mdp_dataset_builder.init_args(),
            "n_steps": self.n_steps,
            # model internal hyperparams
            **self._get_model_hyperparams(),
            "use_gpu": self.model._impl.device
        }

    def _save_model(self, path: str) -> None:
        self.logger.info('-- Saving model to %s', path)
        self.model.save_model(path)

    def _load_model(self, path: str) -> None:
        self.logger.info('-- Loading model from %s', path)
        self.model.load_model(path)

    def _get_model_hyperparams(self) -> Dict[str, Any]:
        """Get model hyperparams as dictionary.
        NB: The code is taken from a `d3rlpy.base.save_config(logger)` method as
        there's no method to just return such params without saving them.
        """
        assert self.model._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        config = LearnableConfigWithShape(
            observation_shape=self.model.impl.observation_shape,
            action_size=self.model.impl.action_size,
            config=self.model.config,
        )
        config = config.serialize_to_dict()
        config.update(config["config"]["params"])
        for key_to_delete in ["observation_shape", "action_size", "config"]:
            config.pop(key_to_delete)

        return config

    def _serialize_policy(self) -> bytes:
        # store using temporary file and immediately read serialized version
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # noinspection PyProtectedMember
            self.model.save_policy(tmp.name)
            with open(tmp.name, 'rb') as policy_file:
                return policy_file.read()

    @staticmethod
    def _deserialize_policy(policy: bytes) -> torch.jit.ScriptModule:
        with io.BytesIO(policy) as buffer:
            return torch.jit.load(buffer, map_location=torch.device('cpu'))

    @staticmethod
    def _predict_relevance_with_policy(
            policy: torch.jit.ScriptModule, items: np.ndarray
    ) -> np.ndarray:
        items = torch.from_numpy(items).float().cpu()
        with torch.no_grad():
            return policy.forward(items).numpy()


class MdpDatasetBuilder:
    r"""
    Markov Decision Process Dataset builder.
    This class transforms datasets with user logs, which is natural for recommender systems,
    to datasets consisting of users' decision-making session logs, which is natural for RL methods.

    Args:
        top_k (int): the number of top user items to learn predicting.
        action_randomization_scale (float): the scale of action randomization gaussian noise.
    """
    logger: logging.Logger
    top_k: int
    action_randomization_scale: float

    def __init__(self, top_k: int, action_randomization_scale: float = 1e-3):
        self.logger = logging.getLogger("replay")
        self.top_k = top_k
        # cannot set zero scale as then d3rlpy will treat transitions as discrete
        assert action_randomization_scale > 0
        self.action_randomization_scale = action_randomization_scale

    def build(self, log: SparkDataFrame) -> MDPDataset:
        """Builds and returns MDP dataset from users' log."""

        start_time = timer()
        # reward top-K watched movies with 1, the others - with 0
        reward_condition = sf.row_number().over(
            Window
            .partitionBy('user_idx')
            .orderBy([sf.desc('relevance'), sf.desc('timestamp')])
        ) <= self.top_k

        # every user has his own episode (the latest item is defined as terminal)
        terminal_condition = sf.row_number().over(
            Window
            .partitionBy('user_idx')
            .orderBy(sf.desc('timestamp'))
        ) == 1

        user_logs = (
            log
            .withColumn("reward", sf.when(reward_condition, sf.lit(1)).otherwise(sf.lit(0)))
            .withColumn("terminal", sf.when(terminal_condition, sf.lit(1)).otherwise(sf.lit(0)))
            .withColumn(
                "action",
                sf.col("relevance").cast("float") + sf.randn() * self.action_randomization_scale
            )
            .orderBy(['user_idx', 'timestamp'], ascending=True)
            .select(['user_idx', 'item_idx', 'action', 'reward', 'terminal'])
            .toPandas()
        )
        train_dataset = MDPDataset(
            observations=np.array(user_logs[['user_idx', 'item_idx']]),
            actions=user_logs['action'].to_numpy()[:, None],
            rewards=user_logs['reward'].to_numpy(),
            terminals=user_logs['terminal'].to_numpy()
        )

        prepare_time = timer() - start_time
        self.logger.info('-- Building MDP dataset took %.2f seconds', prepare_time)
        return train_dataset

    # pylint: disable=missing-function-docstring
    def init_args(self):
        return {
            "top_k": self.top_k,
            "action_randomization_scale": self.action_randomization_scale,
        }
