from datetime import timedelta
import itertools
import socket
from typing import Optional, Tuple, Dict, Any

from lightfm import LightFM
from lightfm._lightfm_fast import CSRMatrix, FastLightFM, fit_bpr, fit_warp
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as sf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StructField,
    StructType,
)
try:
    import pygloo.dist.pygloo as pgl
except ModuleNotFoundError:
    import pygloo as pgl
from replay.models.base_rec import PartialFitMixin, HybridRecommender
from replay.models.hnswlib import HnswlibMixin
from replay.session_handler import State
from replay.utils import unionify
from scipy.sparse import csr_matrix, coo_matrix
import torch.distributed as dist

LOSSES = ['warp', 'bpr']
CYTHON_DTYPE = np.float32
PORT = 21235
REC_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("relevance", DoubleType()),
    ]
)


class LightFMWrap(HybridRecommender, PartialFitMixin, HnswlibMixin):
    def __init__(
            self,
            no_components: int = 10,
            learning_schedule: str = "adagrad",
            loss: str = "warp",
            learning_rate: float = 0.05,
            rho: float = 0.95,
            epsilon: float = 1e-6,
            # item_alpha=0.0,
            # user_alpha=0.0,
            max_sampled: int = 100,
            random_state: Optional[int] = None,
            tcp_port: int = PORT,
            connection_timeout: int = 30,
            hnswlib_params: Optional[dict] = None,
            num_epochs: int = 30,
    ):
        self.no_components = no_components
        self.learning_schedule = learning_schedule
        self.loss = loss
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.max_sampled = max_sampled
        self.random_state = random_state
        self.tcp_port = tcp_port
        self.connection_timeout = connection_timeout
        self._hnswlib_params = hnswlib_params

        self.model = None
        self.world_size = None
        self._num_epochs = num_epochs

        self._check_parameters()

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, value):
        if value <= 0:
            raise ValueError("Defined number of training epochs must be positive.")
        self._num_epochs = value

    @property
    def _use_ann(self) -> bool:
        return self._hnswlib_params is not None

    def _get_vectors_to_build_ann(self, log: DataFrame) -> DataFrame:
        item_vectors, _ = self.get_features(
            log.select("item_idx").distinct()
        )
        return item_vectors

    def _get_vectors_to_infer_ann_inner(
            self, log: DataFrame, users: DataFrame
    ) -> DataFrame:
        user_vectors, _ = self.get_features(users)
        return user_vectors

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": "user_factors",
            "params": self._hnswlib_params,
            "index_dim": self.model.no_components + 1,
        }

    def _get_ann_build_params(self, log: DataFrame) -> Dict[str, Any]:
        self.num_elements = log.select("item_idx").distinct().count()
        return {
            "features_col": "item_factors",
            "params": self._hnswlib_params,
            "dim": self.model.no_components + 1,
            "num_elements": self.num_elements,
            "id_col": "item_idx",
        }

    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame] = None
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        entity = "user" if "user_idx" in ids.columns else "item"
        if features:
            self._convert_features_to_csr(features)
        _biases, _embeddings = self.model.__getattribute__(f'get_{entity}_representations')(features=features)

        # Concatenate embeddings and biases to get vector representations of <entities>
        representations = np.concatenate([_embeddings, _biases.reshape(-1, 1)], axis=1)

        def _representations(representations_arr: np.ndarray):
            for entity_idx in range(representations_arr.shape[0]):
                yield (entity_idx, representations_arr[entity_idx].tolist())

        lightfm_factors = State().session.createDataFrame(
            _representations(representations),
            schema=StructType(
                [
                    StructField(f"{entity}_idx", IntegerType()),
                    StructField(f"{entity}_factors", ArrayType(DoubleType()))
                ]
            ),
        )
        return (
            lightfm_factors.join(ids, how="right", on=f"{entity}_idx"),
            self.model.no_components + 1,
        )

    @property
    def _init_args(self):
        return {
            "no_components": self.no_components,
            "learning_schedule": self.learning_schedule,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "rho": self.rho,
            "epsilon": self.epsilon,
            # "item_alpha": self.item_alpha,
            # "user_alpha": self.user_alpha,
            "max_sampled": self.max_sampled,
            "random_state": self.random_state,
            "tcp_port": self.tcp_port,
            "connection_timeout": self.connection_timeout,
        }

    def _predict_selected_pairs(
        self,
        pairs: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ):
        def predict_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
            pandas_df["relevance"] = model.predict(
                user_ids=pandas_df["user_idx"].to_numpy(),
                item_ids=pandas_df["item_idx"].to_numpy(),
                item_features=csr_item_features,
                user_features=csr_user_features,
            )
            return pandas_df

        model = self.model

        # TODO check assertions
        if self.can_predict_cold_users and user_features is None:
            raise ValueError("User features are missing for predict")
        if self.can_predict_cold_items and item_features is None:
            raise ValueError("Item features are missing for predict")

        csr_item_features = self._convert_features_to_csr(item_features) # pairs.select("item_idx").distinct(),
        csr_user_features = self._convert_features_to_csr(user_features) # pairs.select("user_idx").distinct(),

        return pairs.groupby("user_idx").applyInPandas(
            predict_by_user, REC_SCHEMA
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
        # TODO question: do we need to get top k? optimization
        users = users.distinct()
        items = items.distinct()
        pairs = users.crossJoin(items)
        predict = self._predict_selected_pairs(pairs, user_features, item_features)
        if filter_seen_items:
            return predict.join(
                log,
                [predict.user_idx == log.user_idx, predict.item_idx == log.item_idx],
                "leftanti"
            )
        return predict

    def _initialize_world_size_and_threads(self):
        sc = SparkSession.getActiveSession().sparkContext
        self.world_size = sc.defaultParallelism
        self.num_threads = 1  # TODO

    def _check_parameters(self):
        if self.loss not in LOSSES:
            raise NotImplementedError(
                "Only `warp`, `bpr` losses can be used."
            )

    def _initialize_model(self):
        self.model = LightFM(
            no_components=self.no_components,
            learning_schedule=self.learning_schedule,
            loss=self.loss,
            learning_rate=self.learning_rate,
            rho=self.rho,
            epsilon=self.epsilon,
            # item_alpha=0.0,
            # user_alpha=0.0,
            max_sampled=self.max_sampled,
            random_state=self.random_state,
        )

    def _convert_features_to_csr(self, features: Optional[DataFrame] = None) -> Optional[csr_matrix]:
        """
        returns
        features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        """
        # TODO preprocessing of features if not none
        if not features:
            return
        raise NotImplementedError

    def _fit(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> None:
        self._initialize_model()
        self._fit_partial(log, user_features, item_features)

    def _get_num_users_and_items(self, log: DataFrame, previous_log: Optional[DataFrame] = None):
        interactions = unionify(log, previous_log)
        num_users = interactions.agg({"user_idx": "max"}).collect()[0][0]
        num_items = interactions.agg({"item_idx": "max"}).collect()[0][0]
        return num_users + 1, num_items + 1

    @staticmethod
    def _get_host_ip():
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address

    def _rdd_to_csr(self, partition_interactions: itertools.chain, num_users: int, num_items: int):
        user_ids, item_ids, relevance = [], [], []
        for row in partition_interactions:
            user_ids.append(row.user_idx)
            item_ids.append(row.item_idx)
            relevance.append(row.relevance)

        csr = csr_matrix(
            (relevance, (user_ids, item_ids)),
            shape=(num_users, num_items),
        )
        return csr

    def _initialize_local_state(self) -> None:
        """ Create local copy of the model states. """
        self.local_item_feature_gradients = self.model.item_embedding_gradients.copy()
        self.local_item_feature_momentum = self.model.item_embedding_momentum.copy()
        self.local_item_bias_gradients = self.model.item_bias_gradients.copy()
        self.local_item_bias_momentum = self.model.item_bias_momentum.copy()
        self.local_user_feature_gradients = self.model.user_embedding_gradients.copy()
        self.local_user_feature_momentum = self.model.user_embedding_momentum.copy()
        self.local_user_bias_gradients = self.model.user_bias_gradients.copy()
        self.local_user_bias_momentum = self.model.user_bias_momentum.copy()

    def _reinitialize_embeddings(
            self,
            log: DataFrame,
            previous_log: DataFrame,
            num_users: int,
            num_items: int,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> None:
        # Get number of users and items presented in previous log
        num_users_previous_log, num_items_previous_log = self._get_num_users_and_items(previous_log)
        if (not self.model.user_embeddings.shape[0] != num_users_previous_log) \
                or (not self.model.item_embeddings.shape[0] == num_items_previous_log):
            raise ValueError(
                'Number of user/items features in trained model is not equal to number of users/items in previous log.'
            )

        # Get number of users and items added in log
        new_users = num_users - num_users_previous_log
        new_items = num_items - num_items_previous_log

        # Get number of non-I users` and items` features in previous log and log
        num_user_extra_features_previous_log = self.model.user_embeddings.shape[0] - num_users_previous_log
        num_item_extra_features_previous_log = self.model.item_embeddings.shape[0] - num_items_previous_log

        num_user_extra_features_log = user_features.shape[1] - num_users
        num_item_extra_features_log = item_features.shape[1] - num_items

        if not (num_user_extra_features_previous_log == num_user_extra_features_log
                and num_item_extra_features_previous_log == num_item_extra_features_log):
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
            return np.concatenate((model_array[:dissect_at], added_array, model_array[:dissect_at]))

        item_updates = [
            ('item_embeddings', new_item_embeddings),
            ('item_embedding_gradients', new_item_embedding_gradients),
            ('item_embedding_momentum', new_item_embedding_momentum),
            ('item_biases', new_item_biases),
            ('item_bias_gradients', new_item_bias_gradients),
            ('item_bias_momentum', new_item_bias_momentum),
        ]
        for attribute_name, value in item_updates:
            updated_value = _concat_arrays(self.model.__getattribute__(attribute_name), value, num_items_previous_log)
            self.model.__setattr__(attribute_name, updated_value)

        user_updates = [
            ('user_embeddings', new_user_embeddings),
            ('user_embedding_gradients', new_user_embedding_gradients),
            ('user_embedding_momentum', new_user_embedding_momentum),
            ('user_biases', new_user_biases),
            ('user_bias_gradients', new_user_bias_gradients),
            ('user_bias_momentum', new_user_bias_momentum),
        ]
        for attribute_name, value in user_updates:
            updated_value = _concat_arrays(self.model.__getattribute__(attribute_name), value, num_users_previous_log)
            self.model.__setattr__(attribute_name, updated_value)

    def _fit_partial(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            previous_log: Optional[DataFrame] = None,
    ) -> None:
        if not self.model:
            self._initialize_model()  # TODO question: can the previous log be passed in newly initialized model?
        self._initialize_world_size_and_threads()  # TODO: change to function

        log = log.select("user_idx", "item_idx", sf.lit(1).alias("relevance"))
        interactions = log.repartition(self.world_size, "user_idx")

        n_users, n_items = self._get_num_users_and_items(log=interactions, previous_log=previous_log)

        user_features = self._convert_features_to_csr(user_features)
        item_features = self._convert_features_to_csr(item_features)

        (user_features, item_features) = self.model._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

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
                # TODO question:
                #  is it possible?
                raise ValueError('Passed previous log will not affect training as model has not been trained yet.')

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

        host = self._get_host_ip()
        real_store = dist.TCPStore(
            host,
            self.tcp_port,
            self.world_size + 1,
            True,
            timedelta(seconds=self.connection_timeout)
        )

        def udf_to_map_on_interactions_with_index(p_idx, partition_interactions):
            # Initialize pygloo context
            context = pgl.rendezvous.Context(p_idx, self.world_size)
            local_ip = self._get_host_ip()
            attr = pgl.transport.tcp.attr(local_ip)
            dev = pgl.transport.tcp.CreateDevice(attr)
            local_store = dist.TCPStore(
                host,
                self.tcp_port,
                self.world_size + 1,
                False,
                timedelta(seconds=self.connection_timeout)
            )
            store = pgl.rendezvous.CustomStore(local_store)
            context.connectFullMesh(store, dev)
            self.gloo_context = context

            # Prapare data
            interactions = self._rdd_to_csr(partition_interactions, num_users=n_users, num_items=n_items)
            interactions = interactions.tocoo()

            if interactions.dtype != CYTHON_DTYPE:
                interactions.data = interactions.data.astype(CYTHON_DTYPE)

            # Initialize sample weights
            sample_weight_data = self.model._process_sample_weight(interactions, sample_weight=None)
            for input_data in (
                    interactions.data,
                    sample_weight_data,
            ):
                self.model._check_input_finite(input_data)

            # Get local interactions partition in COO sparse matrix format
            interactions_part = coo_matrix(
                (interactions.data, (interactions.row, interactions.col)),
                shape=(n_users, n_items),
            )

            # Copy model states to executors
            self._initialize_local_state()

            # Each Spark executor runs on interaction matrix partition
            for _ in self.model._progress(self._num_epochs, verbose=False):
                self._run_epoch_spark(
                    item_features,
                    user_features,
                    interactions_part,
                    sample_weight_data,
                    self.num_threads,
                    self.model.loss,
                )
                self.model._check_finite()

            if p_idx == 0:
                self.gloo_context = None
                yield self.model

        self.model = interactions.rdd.mapPartitionsWithIndex(udf_to_map_on_interactions_with_index).collect()[0]

    def copy_representations_for_update(self) -> None:
        """ Create local copy of the item and user representations. """
        self.local_item_features = self.model.item_embeddings.copy()
        self.local_item_biases = self.model.item_biases.copy()
        self.local_user_features = self.model.user_embeddings.copy()
        self.local_user_biases = self.model.user_biases.copy()

    def _get_lightfm_data(self) -> FastLightFM:
        """ Create FastLightFM class from the states to run update. """

        lightfm_data = FastLightFM(
            self.local_item_features,
            self.local_item_feature_gradients,
            self.local_item_feature_momentum,
            self.local_item_biases,
            self.local_item_bias_gradients,
            self.local_item_bias_momentum,
            self.local_user_features,
            self.local_user_feature_gradients,
            self.local_user_feature_momentum,
            self.local_user_biases,
            self.local_user_bias_gradients,
            self.local_user_bias_momentum,
            self.model.no_components,
            int(self.model.learning_schedule == "adadelta"),
            self.model.learning_rate,
            self.model.rho,
            self.model.epsilon,
            self.model.max_sampled,
        )

        return lightfm_data

    def _get_update_delta_after_fit(self):
        """ Extract initial representation values to get delta from update. """

        self.local_item_features -= self.model.item_embeddings
        self.local_item_biases -= self.model.item_biases
        self.local_user_features -= self.model.user_embeddings
        self.local_user_biases -= self.model.user_biases

    def _reduce_states_on_workers(self):
        """ Perform AllReduce operation summing up representations and averaging the optimization parameters. """

        sum_attributes = (
            "local_user_features",
            "local_user_biases",
        )

        average_attributes = (
            "local_item_features",
            "local_item_biases",
            "local_item_feature_gradients",
            "local_item_feature_momentum",
            "local_item_bias_gradients",
            "local_item_bias_momentum",
            "local_user_feature_gradients",
            "local_user_feature_momentum",
            "local_user_bias_gradients",
            "local_user_bias_momentum",
        )

        for attr_name in sum_attributes + average_attributes:
            sendbuf = self.__getattribute__(attr_name)
            recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
            sendptr = sendbuf.ctypes.data
            recvptr = recvbuf.ctypes.data
            data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
            datatype = pgl.glooDataType_t.glooFloat32

            pgl.allreduce(
                self.gloo_context,
                sendptr,
                recvptr,
                data_size,
                datatype,
                pgl.ReduceOp.SUM,
                pgl.allreduceAlgorithm.RING
            )
            self.__setattr__(attr_name, recvbuf)

        for attr_name in average_attributes:
            attr_value = self.__getattribute__(attr_name)
            self.__setattr__(attr_name, attr_value / self.world_size)

    def _update_model_with_reduced_data(self):
        """ Updates model state after parallel operations. """

        self.model.item_embeddings += self.local_item_features
        self.model.item_embedding_gradients = self.local_item_feature_gradients
        self.model.item_embedding_momentum = self.local_item_feature_momentum
        self.model.item_biases += self.local_item_biases
        self.model.item_bias_gradients = self.local_item_bias_gradients
        self.model.item_bias_momentum = self.local_item_bias_momentum
        self.model.user_embeddings += self.local_user_features
        self.model.user_embedding_gradients = self.local_user_feature_gradients
        self.model.user_embedding_momentum = self.local_user_feature_momentum
        self.model.user_biases += self.local_user_biases
        self.model.user_bias_gradients = self.local_user_bias_gradients
        self.model.user_bias_momentum = self.local_user_bias_momentum

    def _run_epoch_spark(
            self,
            item_features,
            user_features,
            interactions,
            sample_weight,
            num_threads,
            loss,
    ):
        positives_lookup = CSRMatrix(
            self.model._get_positives_lookup_matrix(interactions)
        )  # only for ("warp", "bpr", "warp-kos")

        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        self.model.random_state.shuffle(shuffle_indices)

        # Get representations copies from the local model
        self.copy_representations_for_update()
        lightfm_data = self._get_lightfm_data()

        if loss == "warp":
            # Run updates on the model state copy
            fit_warp(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.model.learning_rate,
                self.model.item_alpha,  # TODO regulatization
                self.model.user_alpha,
                num_threads,
                self.model.random_state,
            )
        elif loss == "bpr":
            fit_bpr(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.model.learning_rate,
                self.model.item_alpha,  # TODO regulatization
                self.model.user_alpha,
                num_threads,
                self.model.random_state,
            )
        else:
            raise NotImplementedError(
                "Only `warp`, `bpr` losses are available by the moment"
            )

        # Get embeddings deltas before reduction
        self._get_update_delta_after_fit()
        # Perform AllReduce reduction on local states
        self._reduce_states_on_workers()
        # Update local models with common model states
        self._update_model_with_reduced_data()
