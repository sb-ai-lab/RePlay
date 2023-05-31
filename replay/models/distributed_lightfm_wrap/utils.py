import itertools
import socket
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from lightfm import LightFM
from lightfm._lightfm_fast import (
    CSRMatrix,
    FastLightFM,
    fit_bpr,
    fit_logistic,
    fit_warp,
    fit_warp_kos,
)
import pygloo as pgl

from scipy.sparse import csr_matrix


@dataclass
class ModelFeatures:
    """ Model representations. """

    item_features: np.ndarray
    item_biases: np.ndarray
    user_features: np.ndarray
    user_biases: np.ndarray


# pylint: disable=too-many-instance-attributes, unnecessary-dunder-call
@dataclass
class ModelTrainingFeatures:
    """ Model features used on training. """

    item_feature_gradients: np.ndarray
    item_feature_momentum: np.ndarray
    item_bias_gradients: np.ndarray
    item_bias_momentum: np.ndarray
    user_feature_gradients: np.ndarray
    user_feature_momentum: np.ndarray
    user_bias_gradients: np.ndarray
    user_bias_momentum: np.ndarray

    user_alpha: np.ndarray
    item_alpha: np.ndarray


# pylint: disable=too-many-instance-attributes, too-many-arguments
class LightFMTraining:
    """LightFMTraining class used in distributed LightFM to perform training steps and results synchronization.

    To perform a LightFM model training in a distributed fashion, each Spark executor is initialized as a Gloo worker.
     The interation matrix is divided between executors. Additionally, OpenMP threads are initialized according to
     the number of Spark executor cores.
     Before the training begins, interaction matrix is partitioned by `user_idx`, model states (item and user embeddings
     and biases, along with training features) are copied to the Spark executors. Then, the Gloo context is initialized,
     each executor is registered as the Gloo worker, each of which will train the model on the part of the interaction
     matrix.
     During the training, on each epoch, model`s weights and user/item representations updates are performed separately
     by Gloo workers. At the end of each epoch, worker collects the updates and AllReduce collective operation is
     performed to synchronize the results with other Gloo workers.

    :param model: LightFM model instance
    :param world_size: collective communication world size - number of Gloo workers
    :param num_threads: number of threads for OpenMP parallelization within an executor
    """

    def __init__(self, model: LightFM, world_size: int, num_threads: int):
        self.model = model
        self.world_size = world_size
        self.num_threads = num_threads

    def initialize_local_state(self) -> ModelTrainingFeatures:
        """ Create local copy of the model states. """
        return ModelTrainingFeatures(
            item_feature_gradients=self.model.item_embedding_gradients.copy(),
            item_feature_momentum=self.model.item_embedding_momentum.copy(),
            item_bias_gradients=self.model.item_bias_gradients.copy(),
            item_bias_momentum=self.model.item_bias_momentum.copy(),
            user_feature_gradients=self.model.user_embedding_gradients.copy(),
            user_feature_momentum=self.model.user_embedding_momentum.copy(),
            user_bias_gradients=self.model.user_bias_gradients.copy(),
            user_bias_momentum=self.model.user_bias_momentum.copy(),
            user_alpha=self.model.user_alpha,
            item_alpha=self.model.item_alpha,
        )

    @staticmethod
    def rdd_to_csr(
        partition_interactions: itertools.chain, num_users: int, num_items: int
    ):
        """ Convert Spark RDD into scipy CSR matrix."""
        user_ids, item_ids, relevance = [], [], []
        for row in partition_interactions:
            user_ids.append(row.user_idx)
            item_ids.append(row.item_idx)
            relevance.append(row.relevance)

        csr = csr_matrix(
            (relevance, (user_ids, item_ids)), shape=(num_users, num_items),
        )
        return csr

    @staticmethod
    def get_host_ip():
        """ Get the IP of the host. """
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address

    def _copy_representations_for_update(self) -> ModelFeatures:
        """ Create local copy of the item and user representations. """
        return ModelFeatures(
            item_features=self.model.item_embeddings.copy(),
            item_biases=self.model.item_biases.copy(),
            user_features=self.model.user_embeddings.copy(),
            user_biases=self.model.user_biases.copy(),
        )

    def _get_lightfm_data(
        self, features: ModelFeatures, training_features: ModelTrainingFeatures
    ) -> FastLightFM:
        """ Create FastLightFM class from the states to run update. """

        lightfm_data = FastLightFM(
            features.item_features,
            training_features.item_feature_gradients,
            training_features.item_feature_momentum,
            features.item_biases,
            training_features.item_bias_gradients,
            training_features.item_bias_momentum,
            features.user_features,
            training_features.user_feature_gradients,
            training_features.user_feature_momentum,
            features.user_biases,
            training_features.user_bias_gradients,
            training_features.user_bias_momentum,
            self.model.no_components,
            int(self.model.learning_schedule == "adadelta"),
            self.model.learning_rate,
            self.model.rho,
            self.model.epsilon,
            self.model.max_sampled,
        )

        return lightfm_data

    def run_epoch_spark(
        self,
        item_features,
        user_features,
        interactions,
        sample_weight,
        gloo_context,
        training_features: ModelTrainingFeatures,
    ):
        """ Run an epoch of training. """
        if self.model.loss in ("warp", "bpr", "warp-kos",):
            positives_lookup = CSRMatrix(
                self.model._get_positives_lookup_matrix(interactions)
            )

        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        self.model.random_state.shuffle(shuffle_indices)

        # Get representations copies from the local model
        local_features = self._copy_representations_for_update()
        lightfm_data = self._get_lightfm_data(
            features=local_features, training_features=training_features
        )

        if self.model.loss == "warp":
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
                training_features.item_alpha,
                training_features.user_alpha,
                self.num_threads,
                self.model.random_state,
            )
        elif self.model.loss == "bpr":
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
                training_features.item_alpha,
                training_features.user_alpha,
                self.num_threads,
                self.model.random_state,
            )
        elif self.model.loss == "warp-kos":
            fit_warp_kos(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                shuffle_indices,
                lightfm_data,
                self.model.learning_rate,
                training_features.item_alpha,
                training_features.user_alpha,
                self.model.k,
                self.model.n,
                self.num_threads,
                self.model.random_state,
            )
        elif self.model.loss == "logistic":
            fit_logistic(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.model.learning_rate,
                training_features.item_alpha,
                training_features.user_alpha,
                self.num_threads,
            )
        else:
            raise NotImplementedError(
                "Only `warp`, `bpr` losses are available by the moment"
            )

        # Get embeddings deltas before reduction
        local_features = self._get_update_delta_after_fit(local_features)
        # Perform AllReduce reduction on local states
        local_features, training_features = self._reduce_states_on_workers(
            gloo_context, local_features, training_features
        )
        # Update local models with common model states
        self._update_model_with_reduced_data(local_features, training_features)

    def _get_update_delta_after_fit(
        self, local_features: ModelFeatures
    ) -> ModelFeatures:
        """ Extract initial representation values to get delta from update. """
        local_features.item_features -= self.model.item_embeddings
        local_features.item_biases -= self.model.item_biases
        local_features.user_features -= self.model.user_embeddings
        local_features.user_biases -= self.model.user_biases
        return local_features

    def _reduce_states_on_workers(
        self,
        gloo_context,
        features: ModelFeatures,
        training_features: ModelTrainingFeatures,
    ) -> Tuple[ModelFeatures, ModelTrainingFeatures]:
        """ Perform AllReduce operation on representations and optimization parameters. """

        def reduce_(
            item: Union[ModelTrainingFeatures, ModelFeatures],
            attr_name: str,
            operation: str = "sum",
        ) -> Union[ModelTrainingFeatures, ModelFeatures]:
            sendbuf = item.__getattribute__(attr_name)
            recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
            sendptr = sendbuf.ctypes.data
            recvptr = recvbuf.ctypes.data
            data_size = (
                sendbuf.size
                if isinstance(sendbuf, np.ndarray)
                else sendbuf.numpy().size
            )
            datatype = pgl.glooDataType_t.glooFloat32

            pgl.allreduce(
                gloo_context,
                sendptr,
                recvptr,
                data_size,
                datatype,
                pgl.ReduceOp.SUM,
                pgl.allreduceAlgorithm.RING,
            )

            item.__setattr__(attr_name, recvbuf)
            if operation == "avg":
                attr_value = item.__getattribute__(attr_name)
                item.__setattr__(attr_name, attr_value / self.world_size)

            return item

        avg_attributes_features = (
            "item_features",
            "item_biases",
        )
        sum_attributes_features = (
            "user_features",
            "user_biases",
        )
        avg_attributes_training_features = (
            "item_feature_gradients",
            "item_feature_momentum",
            "item_bias_gradients",
            "item_bias_momentum",
            "user_feature_gradients",
            "user_feature_momentum",
            "user_bias_gradients",
            "user_bias_momentum",
        )

        for sum_attributes_feat in sum_attributes_features:
            features = reduce_(
                features, attr_name=sum_attributes_feat, operation="sum"
            )

        for avg_attributes_feat in avg_attributes_features:
            features = reduce_(
                features, attr_name=avg_attributes_feat, operation="avg"
            )

        for avg_attributes_tr_feat in avg_attributes_training_features:
            training_features = reduce_(
                training_features,
                attr_name=avg_attributes_tr_feat,
                operation="avg",
            )

        return features, training_features

    def _update_model_with_reduced_data(
        self,
        features: ModelFeatures,
        training_features: ModelTrainingFeatures,
    ):
        """ Updates model state after parallel operations. """

        self.model.item_embeddings += features.item_features
        self.model.item_embedding_gradients = (
            training_features.item_feature_gradients
        )
        self.model.item_embedding_momentum = (
            training_features.item_feature_momentum
        )
        self.model.item_biases += features.item_biases
        self.model.item_bias_gradients = training_features.item_bias_gradients
        self.model.item_bias_momentum = training_features.item_bias_momentum

        self.model.user_embeddings += features.user_features
        self.model.user_embedding_gradients = (
            training_features.user_feature_gradients
        )
        self.model.user_embedding_momentum = (
            training_features.user_feature_momentum
        )
        self.model.user_biases += features.user_biases
        self.model.user_bias_gradients = training_features.user_bias_gradients
        self.model.user_bias_momentum = training_features.user_bias_momentum
