from typing import Any, Dict, Optional, Tuple

import numba as nb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from replay.experimental.models.base_neighbour_rec import NeighbourRec
from replay.experimental.utils.session_handler import State
from replay.models.extensions.ann.index_builders.base_index_builder import IndexBuilder
from replay.utils import SparkDataFrame


@nb.njit(parallel=True)
def _main_iteration(
    inv_matrix,
    p_x,
    mat_b,
    mat_c,
    mat_gamma,
    rho,
    eps_abs,
    eps_rel,
    lambda_1,
    items_count,
    threshold,
    multiplicator,
):  # pragma: no cover
    # calculate mat_b
    mat_b = p_x + np.dot(inv_matrix, rho * mat_c - mat_gamma)
    vec_gamma = np.diag(mat_b) / np.diag(inv_matrix)
    mat_b -= inv_matrix * vec_gamma

    # calculate mat_c
    prev_mat_c = mat_c
    mat_c = mat_b + mat_gamma / rho
    coef = lambda_1 / rho
    mat_c = np.maximum(mat_c - coef, 0.0) - np.maximum(-mat_c - coef, 0.0)

    # calculate mat_gamma
    mat_gamma += rho * (mat_b - mat_c)

    # calculate residuals
    r_primal = np.linalg.norm(mat_b - mat_c)
    r_dual = np.linalg.norm(-rho * (mat_c - prev_mat_c))
    eps_primal = eps_abs * items_count + eps_rel * max(np.linalg.norm(mat_b), np.linalg.norm(mat_c))
    eps_dual = eps_abs * items_count + eps_rel * np.linalg.norm(mat_gamma)
    if r_primal > threshold * r_dual:
        rho *= multiplicator
    elif threshold * r_primal < r_dual:
        rho /= multiplicator

    return (
        mat_b,
        mat_c,
        mat_gamma,
        rho,
        r_primal,
        r_dual,
        eps_primal,
        eps_dual,
    )


class ADMMSLIM(NeighbourRec):
    """`ADMM SLIM: Sparse Recommendations for Many Users
    <http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf>`_

    This is a modification for the basic SLIM model.
    Recommendations are improved with Alternating Direction Method of Multipliers.
    """

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": None,
        }

    rho: float
    threshold: float = 5
    multiplicator: float = 2
    eps_abs: float = 1.0e-3
    eps_rel: float = 1.0e-3
    max_iteration: int = 100
    _mat_c: np.ndarray
    _mat_b: np.ndarray
    _mat_gamma: np.ndarray
    _search_space = {
        "lambda_1": {"type": "loguniform", "args": [1e-9, 50]},
        "lambda_2": {"type": "loguniform", "args": [1e-9, 5000]},
    }

    def __init__(
        self,
        lambda_1: float = 5,
        lambda_2: float = 5000,
        seed: Optional[int] = None,
        index_builder: Optional[IndexBuilder] = None,
    ):
        """
        :param lambda_1: l1 regularization term
        :param lambda_2: l2 regularization term
        :param seed: random seed
        :param index_builder: `IndexBuilder` instance that adds ANN functionality.
            If not set, then ann will not be used.
        """
        if lambda_1 < 0 or lambda_2 <= 0:
            msg = "Invalid regularization parameters"
            raise ValueError(msg)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.rho = lambda_2
        self.seed = seed
        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)

    @property
    def _init_args(self):
        return {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "seed": self.seed,
        }

    def _fit(
        self,
        log: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,  # noqa: ARG002
        item_features: Optional[SparkDataFrame] = None,  # noqa: ARG002
    ) -> None:
        self.logger.debug("Fitting ADMM SLIM")
        pandas_log = log.select("user_idx", "item_idx", "relevance").toPandas()
        interactions_matrix = csr_matrix(
            (
                pandas_log["relevance"],
                (pandas_log["user_idx"], pandas_log["item_idx"]),
            ),
            shape=(self._user_dim, self._item_dim),
        )
        self.logger.debug("Gram matrix")
        xtx = (interactions_matrix.T @ interactions_matrix).toarray()
        self.logger.debug("Inverse matrix")
        inv_matrix = np.linalg.inv(xtx + (self.lambda_2 + self.rho) * np.eye(self._item_dim))
        self.logger.debug("Main calculations")
        p_x = inv_matrix @ xtx
        mat_b, mat_c, mat_gamma = self._init_matrix(self._item_dim)
        r_primal = np.linalg.norm(mat_b - mat_c)
        r_dual = np.linalg.norm(self.rho * mat_c)
        eps_primal, eps_dual = 0.0, 0.0
        iteration = 0
        while (r_primal > eps_primal or r_dual > eps_dual) and iteration < self.max_iteration:
            iteration += 1
            (
                mat_b,
                mat_c,
                mat_gamma,
                self.rho,
                r_primal,
                r_dual,
                eps_primal,
                eps_dual,
            ) = _main_iteration(
                inv_matrix,
                p_x,
                mat_b,
                mat_c,
                mat_gamma,
                self.rho,
                self.eps_abs,
                self.eps_rel,
                self.lambda_1,
                self._item_dim,
                self.threshold,
                self.multiplicator,
            )
            result_message = (
                f"Iteration: {iteration}. primal gap: "
                f"{r_primal - eps_primal:.5}; dual gap: "
                f" {r_dual - eps_dual:.5}; rho: {self.rho}"
            )
            self.logger.debug(result_message)

        mat_c_sparse = coo_matrix(mat_c)
        mat_c_pd = pd.DataFrame(
            {
                "item_idx_one": mat_c_sparse.row.astype(np.int32),
                "item_idx_two": mat_c_sparse.col.astype(np.int32),
                "similarity": mat_c_sparse.data,
            }
        )
        self.similarity = State().session.createDataFrame(
            mat_c_pd,
            schema="item_idx_one int, item_idx_two int, similarity double",
        )
        self.similarity.cache().count()

    def _init_matrix(self, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Matrix initialization"""
        if self.seed is not None:
            np.random.seed(self.seed)
        mat_b = np.random.rand(size, size)
        mat_c = np.random.rand(size, size)
        mat_gamma = np.random.rand(size, size)
        return mat_b, mat_c, mat_gamma
