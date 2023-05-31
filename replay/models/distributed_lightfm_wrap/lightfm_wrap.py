import logging
import multiprocessing as mp
from datetime import timedelta
from os.path import join
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from lightfm import LightFM
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    DoubleType,
    IntegerType,
    StructField,
    StructType,
)

try:
    import pygloo.dist.pygloo as pgl
except ModuleNotFoundError:
    import pygloo as pgl

import scipy.sparse as sp
import torch.distributed as dist
from sklearn.preprocessing import MinMaxScaler

from replay.constants import REC_SCHEMA
from replay.models.base_rec import HybridRecommender  # , PartialFitMixin
from replay.models.distributed_lightfm_wrap.utils import LightFMTraining
# from replay.models.hnswlib import HnswlibMixin
from replay.ann.ann_mixin import ANNMixin
from replay.ann.index_builders.base_index_builder import IndexBuilder

from replay.session_handler import State
from replay.utils import (
    check_numeric,
    load_pickled_from_parquet,
    save_picklable_to_parquet,
)

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-arguments, unnecessary-dunder-call
class DistributedLightFMWrap(HybridRecommender, ANNMixin):  # , PartialFitMixin
    """Wrapper for distributed version of the LightFM."""

    model_weights = (
        "item_embeddings",
        "item_embedding_gradients",
        "item_embedding_momentum",
        "item_biases",
        "item_bias_gradients",
        "item_bias_momentum",
        "user_embeddings",
        "user_embedding_gradients",
        "user_embedding_momentum",
        "user_biases",
        "user_bias_gradients",
        "user_bias_momentum",
    )
    losses = ["warp", "bpr", "warp-kos", "logistic"]

    @staticmethod
    def _get_entity_name(df: DataFrame):
        return "item" if "item_idx" in df.columns else "user"

    def _get_features(
            self, ids: DataFrame, features: Optional[DataFrame],
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        if self.model is None:
            raise AttributeError("Model has not been fitted yet.")
        entity = self._get_entity_name(ids)
        # entity = "user" if "user_idx" in ids.columns else "item"
        if features:
            features = self._convert_features_to_csr(
                ids.select(f"{entity}_idx").distinct(), features
            )
        _biases, representations = self.model.__getattribute__(
            f"get_{entity}_representations"
        )(features=features)

        def _representations(
                representations_arr: np.ndarray, biases_arr: np.ndarray
        ):
            biases_arr = biases_arr.tolist()
            for entity_idx in range(representations_arr.shape[0]):
                yield entity_idx, representations_arr[
                    entity_idx
                ].tolist(), biases_arr[entity_idx]

        lightfm_factors = State().session.createDataFrame(
            _representations(representations, _biases),
            schema=StructType(
                [
                    StructField(f"{entity}_idx", IntegerType()),
                    StructField(f"{entity}_factors", ArrayType(DoubleType())),
                    StructField(f"{entity}_bias", DoubleType()),
                ]
            ),
        )
        return (
            lightfm_factors.join(ids, how="right", on=f"{entity}_idx"),
            self.model.no_components,
        )

    def _get_vectors_to_build_ann(self, log: DataFrame) -> DataFrame:
        item_vectors, _ = self._get_features(
            log.select("item_idx").distinct(), None
        )
        return item_vectors

    def _get_ann_build_params(self, log: DataFrame) -> Dict[str, Any]:
        self.index_builder.index_params.dim = self.model.no_components
        self.index_builder.index_params.max_elements = log.select("item_idx").distinct().count()
        return {
            "features_col": "item_factors",
            "ids_col": "item_idx",
        }

    def _get_vectors_to_infer_ann_inner(
            self, log: DataFrame, users: DataFrame
    ) -> DataFrame:
        user_vectors, _ = self._get_features(users, None)
        return user_vectors

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        self.index_builder.index_params.dim = self.model.no_components
        return {
            "features_col": "user_factors",
        }

    @staticmethod
    def unionify(df: DataFrame, df_2: Optional[DataFrame] = None) -> DataFrame:
        if df_2 is not None:
            df = df.unionByName(df_2)
        return df

    def _check_parameters(self):
        assert self.loss in self.losses, f"Only `{'`, `'.join(self.losses)}` losses can be used."

    def __init__(
            self,
            no_components: int = 10,
            learning_schedule: str = "adagrad",
            loss: str = "warp",
            learning_rate: float = 0.05,
            rho: float = 0.95,
            epsilon: float = 1e-6,
            k_kos_warp: int = 5,
            n_kos_warp: int = 10,
            max_sampled: int = 100,
            random_state: Optional[int] = None,
            tcp_port: int = 21235,
            connection_timeout: int = 200,
            index_builder: Optional[IndexBuilder] = None,
            num_epochs: int = 30,
            pygloo_timeout_sec: int = 200,
    ):
        """
        :param no_components: the dimensionality of the feature latent embeddings.
        :param learning_schedule: learning schedule used on training. One of (‘adagrad’, ‘adadelta’)
        :param loss: loss function. One of (‘logistic’, ‘bpr’, ‘warp’, ‘warp-kos’)
        :param learning_rate: initial learning rate for the adagrad learning schedule
        :param rho: moving average coefficient for the adadelta learning schedule
        :param epsilon: conditioning parameter for the adadelta learning schedule
        :param k_kos_warp: for k-OS training, the k-th positive example will be selected from the n positive examples
            sampled for every user
        :param n_kos_warp: for k-OS training, maximum number of positives sampled for each update
        :param max_sampled: maximum number of negative samples used during WARP fitting
        :param random_state: the random seed to use when shuffling the data and initializing the parameters
        :param tcp_port: port of TCP store used in collective communications
        :param connection_timeout: TCP store connection timeout in seconds
        :param index_builder: IndexBuilder instance for ANN predictions
        :param num_epochs: number of training epochs
        :param pygloo_timeout_sec: collective operations timeout in seconds
        """
        if loss == "logistic":
            self.logger.warning(
                "Usage of distributed `logistic` loss can be unstable in performance."
            )

        self.no_components = no_components
        self.learning_schedule = learning_schedule
        self.loss = loss
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.k_kos_warp = k_kos_warp
        self.n_kos_warp = n_kos_warp
        self.max_sampled = max_sampled
        self.random_state = random_state
        self.tcp_port = tcp_port
        self.connection_timeout = connection_timeout

        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)
        self._pygloo_timeout_sec = pygloo_timeout_sec
        self._num_epochs = num_epochs

        self.model: Optional[LightFM] = None
        self.world_size: Optional[int] = None
        self.max_seen_user_idx: Optional[int] = None
        self.max_seen_item_idx: Optional[int] = None
        self.num_elements: Optional[int] = None
        self.num_threads: Optional[int] = None
        self.user_feat_scaler: Optional[MinMaxScaler] = None
        self.item_feat_scaler: Optional[MinMaxScaler] = None

        self._check_parameters()

    @property
    def num_epochs(self):
        """ Get number of training epochs."""
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, value):
        if value <= 0:
            raise ValueError(
                "Defined number of training epochs must be positive."
            )
        self._num_epochs = value

    @property
    def pygloo_timeout_sec(self):
        """ Get timeout value in GLOO collective operations."""
        return self._pygloo_timeout_sec

    @pygloo_timeout_sec.setter
    def pygloo_timeout_sec(self, value):
        if value <= 0:
            raise ValueError("Timout should be longer than 0 seconds.")
        self._pygloo_timeout_sec = value

    def _save_model(self, path: str):
        for weight in self.model_weights:
            value_to_save = self.model.__getattribute__(weight)
            self.model.__setattr__(weight, None)
            save_picklable_to_parquet(value_to_save, join(path, weight))

        save_picklable_to_parquet(self.model, join(path, "LightFM_model"))
        save_picklable_to_parquet(
            self.user_feat_scaler, join(path, "user_feat_scaler")
        )
        save_picklable_to_parquet(
            self.item_feat_scaler, join(path, "item_feat_scaler")
        )

        if self._use_ann:
            self._save_index(path)

    def _load_model(self, path: str):
        self.model = load_pickled_from_parquet(join(path, "LightFM_model"))

        for weight in self.model_weights:
            self.model.__setattr__(
                weight, load_pickled_from_parquet(join(path, weight))
            )

        self.user_feat_scaler = load_pickled_from_parquet(
            join(path, "user_feat_scaler")
        )
        self.item_feat_scaler = load_pickled_from_parquet(
            join(path, "item_feat_scaler")
        )

        if self._use_ann:
            self._load_index(path)

    @property
    def _init_args(self) -> dict:
        return {
            "no_components": self.no_components,
            "learning_schedule": self.learning_schedule,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "rho": self.rho,
            "epsilon": self.epsilon,
            "k_kos_warp": self.k_kos_warp,
            "n_kos_warp": self.n_kos_warp,
            "max_sampled": self.max_sampled,
            "random_state": self.random_state,
            "tcp_port": self.tcp_port,
            "connection_timeout": self.connection_timeout,
            "index_builder": self.index_builder.init_meta_as_dict() if self.index_builder else None,
            "num_epochs": self._num_epochs,
            "pygloo_timeout_sec": self._pygloo_timeout_sec,
        }

    def _initialize_world_size_and_threads(self):
        spark_sess = SparkSession.getActiveSession()
        master_addr = spark_sess.conf.get("spark.master")
        if master_addr.startswith("local-cluster"):
            exec_str, cores_str, _ = master_addr[len("local-cluster["): -1].split(",")
            num_executor_instances = int(exec_str)
            num_executor_cores = int(cores_str)
        elif master_addr.startswith("local"):
            cores_str = master_addr[len("local["): -1]
            num_executor_instances = 1
            num_executor_cores = (int(cores_str) if cores_str != "*" else mp.cpu_count())
        else:
            num_executor_instances = int(spark_sess.conf.get("spark.executor.instances", "1"))
            num_executor_cores = int(spark_sess.conf.get("spark.executor.cores", "1"))
        self.world_size = max(num_executor_instances, 1)
        self.num_threads = max(num_executor_cores, 1)

    def _initialize_and_reset_model_state_and_scalers(self):
        # Reset user / item scalers in case if model was initialized earlier
        self.user_feat_scaler = None
        self.item_feat_scaler = None
        # Initialize / reset model states as in case it was trained earlier
        self.model = LightFM(
            no_components=self.no_components,
            learning_schedule=self.learning_schedule,
            loss=self.loss,
            learning_rate=self.learning_rate,
            rho=self.rho,
            epsilon=self.epsilon,
            k=self.k_kos_warp,
            n=self.n_kos_warp,
            max_sampled=self.max_sampled,
            random_state=self.random_state,
        )

    def _fit(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> None:
        self._initialize_and_reset_model_state_and_scalers()
        self._fit_partial(log, user_features, item_features)

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

        users = users.distinct()
        items = items.distinct()
        pairs = users.crossJoin(items)
        predict = self._predict_selected_pairs(pairs, user_features, item_features)
        return predict

    def _predict_selected_pairs(
            self,
            pairs: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        # @sf.pandas_udf("float")
        # def predict_by_user(user_idx: pd.Series, item_idx: pd.Series) -> pd.DataFrame:
        #     return model.predict(
        #         user_ids=user_idx.to_numpy(),
        #         item_ids=item_idx.to_numpy(),
        #         item_features=csr_item_features,
        #         user_features=csr_user_features,
        #     )

        @sf.pandas_udf(FloatType())
        def predict_by_user(user_idx: pd.Series, item_idx: pd.Series) -> pd.Series:
            return pd.Series(model.predict(
                user_ids=user_idx.copy().to_numpy(),
                item_ids=item_idx.copy().to_numpy(),
                item_features=csr_item_features,
                user_features=csr_user_features,
            ))

        model = self.model

        if self.can_predict_cold_users and user_features is None:
            raise ValueError("User features are missing for predict")
        if self.can_predict_cold_items and item_features is None:
            raise ValueError("Item features are missing for predict")

        csr_item_features = self._convert_features_to_csr(
            pairs.select("item_idx").distinct(), item_features
        )
        csr_user_features = self._convert_features_to_csr(
            pairs.select("user_idx").distinct(), user_features
        )

        return pairs.withColumn('relevance', predict_by_user(sf.col("user_idx"), sf.col("item_idx")))

        # return pairs.groupby("user_idx").applyInPandas(
        #     predict_by_user, REC_SCHEMA
        # )

    def _convert_features_to_csr(
            self, entity_ids: DataFrame, features: Optional[DataFrame] = None
    ) -> Optional[sp.csr_matrix]:
        """
        :param entity_ids: user/item_ids from log
        :param features: user/item_features in DataFrame format
        :return: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        """
        if not features:
            return None

        check_numeric(features)

        # entity = "item" if "item_idx" in features.columns else "user"
        entity = self._get_entity_name(features)
        num_seen_entities = self.__getattribute__(f"max_seen_{entity}_idx")
        features = features.join(entity_ids, on=f"{entity}_idx", how="inner")
        max_passed_id = (
                entity_ids.agg({f"{entity}_idx": "max"}).first()[0] + 1
        )
        num_cold_entities = max(
            0, max_passed_id - self.__getattribute__(f"max_seen_{entity}_idx")
        )
        sparse_features = sp.vstack(
            [
                # Identity features
                sp.eye(
                    num_seen_entities, num_seen_entities + num_cold_entities
                ),
                # Empty features
                sp.csr_matrix(
                    np.zeros(
                        (
                            num_cold_entities,
                            num_seen_entities + num_cold_entities,
                        )
                    )
                ),
            ]
        )

        # features to (num_seen_entities + num_cold_entities) x num_features
        feature_names = [
            col_name
            for col_name in features.columns
            if col_name != f"{entity}_idx"
        ]
        features_np = (
            features.select(f"{entity}_idx", *feature_names)
            .toPandas()
            .to_numpy()
        )
        feature_entity_ids = features_np[:, 0]
        features_columns = features_np[:, 1:]
        number_of_features = features_columns.shape[1]
        # Scale down dense features
        scaler_name = f"{entity}_feat_scaler"
        if self.__getattribute__(scaler_name) is None:
            if not features_columns.size:
                raise ValueError(f"features for {entity}s from log are absent")
            self.__setattr__(scaler_name, MinMaxScaler().fit(features_columns))

        if features_columns.size:
            features_dense = self.__getattribute__(scaler_name).transform(
                features_columns
            )

            features = sp.csr_matrix(
                (
                    features_dense.ravel(),
                    (
                        np.repeat(feature_entity_ids, number_of_features),
                        np.tile(
                            np.arange(number_of_features),
                            feature_entity_ids.shape[0],
                        ),
                    ),
                ),
                shape=(
                    num_seen_entities + num_cold_entities,
                    number_of_features,
                ),
            )
        else:
            features = sp.csr_matrix(
                (num_seen_entities + num_cold_entities, number_of_features)
            )
        return sp.hstack([sparse_features, features])

    def _get_num_users_and_items(
            self, log: DataFrame, previous_log: Optional[DataFrame] = None
    ) -> Tuple[int, int]:
        interactions = self.unionify(log, previous_log)
        num_users = interactions.agg({"user_idx": "max"}).first()[0]
        num_items = interactions.agg({"item_idx": "max"}).first()[0]
        return num_users + 1, num_items + 1

    # pylint: disable=unused-argument
    def _reinitialize_embeddings(
            self,
            log: DataFrame,
            previous_log: DataFrame,
            num_users: int,
            num_items: int,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        :param log: training log
        :param previous_log: previous training log
        :param num_users: number of users in log
        :param num_items: number of items in log
        :param user_features: user features in csr_matrix format (num_users x num_user_features)
        :param item_features: item features in csr_matrix format (num_items x num_item_features)
        :return:
        """
        # Get number of users and items presented in previous log
        (
            num_users_previous_log,
            num_items_previous_log,
        ) = self._get_num_users_and_items(previous_log)
        if (
                not self.model.user_embeddings.shape[0] != num_users_previous_log
        ) or (
                not self.model.item_embeddings.shape[0] == num_items_previous_log
        ):
            raise ValueError(
                "Number of user/items features in trained model is not equal"
                "to number of users/items in previous log."
            )

        # Get number of users and items added in log
        new_users = num_users - num_users_previous_log
        new_items = num_items - num_items_previous_log

        # Get number of non-I users` and items` features in previous log and log
        num_user_extra_features_previous_log = (
                self.model.user_embeddings.shape[0] - num_users_previous_log
        )
        num_item_extra_features_previous_log = (
                self.model.item_embeddings.shape[0] - num_items_previous_log
        )

        num_user_extra_features_log = user_features.shape[1] - num_users
        num_item_extra_features_log = item_features.shape[1] - num_items

        if not (
                num_user_extra_features_previous_log == num_user_extra_features_log
                and num_item_extra_features_previous_log
                == num_item_extra_features_log
        ):
            raise ValueError(
                f"Passed number of extra features for user: {num_user_extra_features_log} \n"
                f"Passed number of extra features for item: {num_item_extra_features_log}. \n"
                f"But model was trained on {num_user_extra_features_previous_log} user and "
                f"{num_item_extra_features_previous_log} item features."
            )

        # Initialise new item features.
        new_item_embeddings = (
                (self.model.random_state.rand(new_items, self.no_components) - 0.5)
                / self.no_components
        ).astype(np.float32)
        new_item_embedding_gradients = np.zeros_like(new_item_embeddings)
        new_item_embedding_momentum = np.zeros_like(new_item_embeddings)
        new_item_biases = np.zeros(new_items, dtype=np.float32)
        new_item_bias_gradients = np.zeros_like(new_item_biases)
        new_item_bias_momentum = np.zeros_like(new_item_biases)

        # Initialise new user features.
        new_user_embeddings = (
                (self.model.random_state.rand(new_users, self.no_components) - 0.5)
                / self.no_components
        ).astype(np.float32)
        new_user_embedding_gradients = np.zeros_like(new_user_embeddings)
        new_user_embedding_momentum = np.zeros_like(new_user_embeddings)
        new_user_biases = np.zeros(new_users, dtype=np.float32)
        new_user_bias_gradients = np.zeros_like(new_user_biases)
        new_user_bias_momentum = np.zeros_like(new_user_biases)

        def _concat_arrays(model_array: np.ndarray, added_array: np.ndarray, dissect_at: int):
            return np.concatenate(
                (
                    model_array[:dissect_at],
                    added_array,
                    model_array[:dissect_at],
                )
            )

        def _update_values(entity_updates: List[Tuple[str, np.ndarray]], num_entities_previous_log: int) -> None:
            for attribute_name, value in entity_updates:
                updated_value = _concat_arrays(
                    self.model.__getattribute__(attribute_name),
                    value,
                    num_entities_previous_log,
                )
                self.model.__setattr__(attribute_name, updated_value)

        item_updates = [
            ("item_embeddings", new_item_embeddings),
            ("item_embedding_gradients", new_item_embedding_gradients),
            ("item_embedding_momentum", new_item_embedding_momentum),
            ("item_biases", new_item_biases),
            ("item_bias_gradients", new_item_bias_gradients),
            ("item_bias_momentum", new_item_bias_momentum),
        ]
        user_updates = [
            ("user_embeddings", new_user_embeddings),
            ("user_embedding_gradients", new_user_embedding_gradients),
            ("user_embedding_momentum", new_user_embedding_momentum),
            ("user_biases", new_user_biases),
            ("user_bias_gradients", new_user_bias_gradients),
            ("user_bias_momentum", new_user_bias_momentum),
        ]

        _update_values(item_updates, num_items_previous_log)
        _update_values(user_updates, num_users_previous_log)
        # for attribute_name, value in item_updates:
        #     updated_value = _concat_arrays(
        #         self.model.__getattribute__(attribute_name),
        #         value,
        #         num_items_previous_log,
        #     )
        #     self.model.__setattr__(attribute_name, updated_value)

        # for attribute_name, value in user_updates:
        #     updated_value = _concat_arrays(
        #         self.model.__getattribute__(attribute_name),
        #         value,
        #         num_users_previous_log,
        #     )
        #     self.model.__setattr__(attribute_name, updated_value)

    # pylint: disable=unused-variable, too-many-statements
    def _fit_partial(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            previous_log: Optional[DataFrame] = None,
    ) -> None:
        if not self.model:
            self._initialize_and_reset_model_state_and_scalers()
        self._initialize_world_size_and_threads()

        self.can_predict_cold_users = user_features is not None
        self.can_predict_cold_items = item_features is not None

        # if user_features is not None:
        #     self.can_predict_cold_users = True
        # if item_features is not None:
        #     self.can_predict_cold_items = True

        log = log.select("user_idx", "item_idx", sf.lit(1).alias("relevance"))
        interactions = log.repartition(self.world_size, "user_idx")

        n_users, n_items = self._get_num_users_and_items(
            log=interactions, previous_log=previous_log
        )
        self.max_seen_user_idx = n_users
        self.max_seen_item_idx = n_items

        user_features = self._convert_features_to_csr(log.select("user_idx").distinct(), user_features)
        item_features = self._convert_features_to_csr(log.select("item_idx").distinct(), item_features)

        (
            user_features,
            item_features,
        ) = self.model._construct_feature_matrices(n_users, n_items, user_features, item_features)

        for input_data in (
                user_features.data,
                item_features.data,
        ):
            self.model._check_input_finite(input_data)

        if previous_log:
            if self.model.item_embeddings:
                self._reinitialize_embeddings(
                    log,
                    previous_log,
                    num_users=n_users,
                    num_items=n_items,
                    user_features=user_features,
                    item_features=item_features,
                )
            else:
                raise ValueError(
                    "Passed previous log will not affect training"
                    "as the model has not been trained yet."
                )

        if self.model.item_embeddings is None:
            self.model._initialize(
                self.model.no_components,
                item_features.shape[1],
                user_features.shape[1],
            )

        if not item_features.shape[1] == self.model.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")
        if not user_features.shape[1] == self.model.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")
        if self.num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        training_instance = LightFMTraining(self.model, self.world_size, self.num_threads)

        host = training_instance.get_host_ip()
        real_store = dist.TCPStore(
            host,
            self.tcp_port,
            self.world_size + 1,
            True,
            timedelta(seconds=self.connection_timeout),
        )

        _tcp_port = self.tcp_port
        _connection_timeout = self.connection_timeout
        _num_epochs = self._num_epochs
        _pygloo_timeout = self._pygloo_timeout_sec

        def udf_to_map_on_interactions_with_index(
                p_idx, partition_interactions
        ):
            # Initialize pygloo context
            context = pgl.rendezvous.Context(p_idx, training_instance.world_size)
            local_ip = training_instance.get_host_ip()
            attr = pgl.transport.tcp.attr(local_ip)
            dev = pgl.transport.tcp.CreateDevice(attr)
            local_store = dist.TCPStore(
                host,
                _tcp_port,
                training_instance.world_size + 1,
                False,
                timedelta(seconds=_connection_timeout),
            )
            store = pgl.rendezvous.CustomStore(local_store)
            context.setTimeout(timedelta(seconds=_pygloo_timeout))
            context.connectFullMesh(store, dev)
            gloo_context = context

            # Prapare data
            interactions = training_instance.rdd_to_csr(partition_interactions, num_users=n_users, num_items=n_items)
            interactions = interactions.tocoo()

            if interactions.dtype != np.float32:
                interactions.data = interactions.data.astype(np.float32)

            # Initialize sample weights
            sample_weight_data = training_instance.model._process_sample_weight(
                interactions, sample_weight=None
            )
            for input_data in (
                    interactions.data,
                    sample_weight_data,
            ):
                training_instance.model._check_input_finite(input_data)

            # Get local interactions partition in COO sparse matrix format
            interactions_part = sp.coo_matrix(
                (interactions.data, (interactions.row, interactions.col)),
                shape=(n_users, n_items),
            )

            # Copy model states to executors
            local_training_features = (
                training_instance.initialize_local_state()
            )

            # Each Spark executor runs on interaction matrix partition
            for _epoch_num in training_instance.model._progress(_num_epochs, verbose=False):
                training_instance.run_epoch_spark(
                    item_features,
                    user_features,
                    interactions_part,
                    sample_weight_data,
                    gloo_context,
                    local_training_features,
                )
                training_instance.model._check_finite()

            if p_idx == 0:
                yield training_instance.model

        self.model = interactions.rdd.mapPartitionsWithIndex(udf_to_map_on_interactions_with_index).first()
