from typing import Optional, Tuple

import numba as nb
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from scipy.sparse import coo_matrix, csr_matrix

from replay.models.base_rec import NeighbourRec
from replay.session_handler import State


# pylint: disable=too-many-arguments, too-many-locals
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

    # calculate mat_с
    prev_mat_c = mat_c
    mat_c = mat_b + mat_gamma / rho
    coef = lambda_1 / rho
    mat_c = np.maximum(mat_c - coef, 0.0) - np.maximum(-mat_c - coef, 0.0)

    # calculate mat_gamma
    mat_gamma += rho * (mat_b - mat_c)

    # calculate residuals
    r_primal = np.linalg.norm(mat_b - mat_c)
    r_dual = np.linalg.norm(-rho * (mat_c - prev_mat_c))
    eps_primal = eps_abs * items_count + eps_rel * max(
        np.linalg.norm(mat_b), np.linalg.norm(mat_c)
    )
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


# pylint: disable=too-many-instance-attributes
class ADMMSLIM(NeighbourRec):
    """`ADMM SLIM: Sparse Recommendations for Many Users
    <http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf>`_

    This is a modification for the basic SLIM model.
    Recommendations are improved with Alternating Direction Method of Multipliers.
    """

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
    ):
        """
        :param lambda_1: l1 regularization term
        :param lambda_2: l2 regularization term
        :param seed: random seed
        """
        if lambda_1 < 0 or lambda_2 <= 0:
            raise ValueError("Invalid regularization parameters")
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.rho = lambda_2
        self.seed = seed

    # pylint: disable=too-many-locals
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.logger.debug("Fitting ADMM SLIM")
        pandas_log = log.select("user_idx", "item_idx", "relevance").toPandas()
        interactions_matrix = csr_matrix(
            (
                pandas_log["relevance"],
                (pandas_log["user_idx"], pandas_log["item_idx"]),
            ),
            shape=(self.users_count, self.items_count),
        )
        self.logger.debug("Gram matrix")
        xtx = (interactions_matrix.T @ interactions_matrix).toarray()
        self.logger.debug("Inverse matrix")
        inv_matrix = np.linalg.inv(
            xtx + (self.lambda_2 + self.rho) * np.eye(self.items_count)
        )
        self.logger.debug("Main calculations")
        p_x = inv_matrix @ xtx
        mat_b, mat_c, mat_gamma = self._init_matrix(self.items_count)
        r_primal = np.linalg.norm(mat_b - mat_c)
        r_dual = np.linalg.norm(self.rho * mat_c)
        eps_primal, eps_dual = 0.0, 0.0
        iteration = 0
        while (
            r_primal > eps_primal or r_dual > eps_dual
        ) and iteration < self.max_iteration:
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
                self.items_count,
                self.threshold,
                self.multiplicator,
            )
            result_message = (
                f"Iteration: {iteration}. primal gap: "
                f"{r_primal - eps_primal:.5}; dual gap: "
                f" {r_dual - eps_dual:.5}; rho: {self.rho}"
            )
            self.logger.info(result_message)

        mat_c_sparse = coo_matrix(mat_c)
        mat_c_pd = pd.DataFrame(
            {
                "item_id_one": mat_c_sparse.row.astype(np.int32),
                "item_id_two": mat_c_sparse.col.astype(np.int32),
                "similarity": mat_c_sparse.data,
            }
        )
        self.similarity = State().session.createDataFrame(mat_c_pd)
        self.similarity.cache()

    def _init_matrix(
        self, size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Matrix initialization"""
        if self.seed is not None:
            np.random.seed(self.seed)
        mat_b = np.random.rand(size, size)  # type: ignore
        mat_c = np.random.rand(size, size)  # type: ignore
        mat_gamma = np.random.rand(size, size)  # type: ignore
        return mat_b, mat_c, mat_gamma
