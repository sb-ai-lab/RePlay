"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from scipy.sparse import csr_matrix, coo_matrix

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.session_handler import State


# pylint: disable=too-many-instance-attributes
class ADMMSLIM(Recommender):
    """`ADMM SLIM: Sparse Recommendations for Many Users
    <http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf>`_

    Улучшение стандартного алгоритма SLIM, направленное на устранение
    некоторых недостатков классического алгоритма. Качество рекомендаций
    повышается за счепт решения задачи с помощью чередующегося метода
    множителей Лагранжа."""

    similarity: DataFrame
    can_predict_cold_users = True
    rho: float
    threshold: float = 5
    multiplicator: float = 2
    eps_abs: float = 1.0e-3
    eps_rel: float = 1.0e-3
    max_iteration: int = 100
    _mat_c: np.ndarray
    _mat_b: np.ndarray
    _mat_gamma: np.ndarray

    def __init__(
        self, lambda_1: float, lambda_2: float, seed: Optional[int] = None
    ):
        """
        :param lambda_1: параметр l1 регуляризации
        :param lambda_2: параметр l2 регуляризации
        :param use_prefit: необходимо ли кэшировать данные
        """
        if lambda_1 < 0 or lambda_2 <= 0:
            raise ValueError("Неверно указаны параметры для регуляризации")
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
        self.logger.debug("Построение модели ADMM SLIM")
        pandas_log = log.select("user_idx", "item_idx", "relevance").toPandas()
        interactions_matrix = csr_matrix(
            (pandas_log.relevance, (pandas_log.user_idx, pandas_log.item_idx)),
            shape=(self.users_count, self.items_count),
        )
        self.logger.debug("Матриица Грама")
        xtx = (interactions_matrix.T @ interactions_matrix).toarray()
        self.logger.debug("Поиск обратной матрицы")
        inv_matrix = np.linalg.inv(
            xtx + (self.lambda_2 + self.rho) * np.eye(self.items_count)
        )
        self.logger.debug("Основной  расчет")
        p_x = inv_matrix @ xtx
        self._mat_b, self._mat_c, self._mat_gamma = self._init_matrix(
            self.items_count
        )
        r_primal = np.linalg.norm(self._mat_b - self._mat_c)
        r_dual = np.linalg.norm(self.rho * self._mat_c)
        eps_primal, eps_dual = 0.0, 0.0
        iteration = 0
        while (
            r_primal > eps_primal or r_dual > eps_dual
        ) and iteration < self.max_iteration:
            iteration += 1
            r_primal, r_dual, eps_primal, eps_dual = self._main_iteration(
                inv_matrix, p_x
            )
            result_message = (
                f"Итерация: {iteration}. primal gap: "
                f"{r_primal - eps_primal:.5}; dual gap: "
                f" {r_dual - eps_dual:.5}; rho: {self.rho}"
            )
            self.logger.debug(result_message)

        mat_c_sparse = coo_matrix(self._mat_c)
        mat_c_pd = pd.DataFrame(
            {
                "item_id_one": mat_c_sparse.row.astype(np.float32),
                "item_id_two": mat_c_sparse.col.astype(np.float32),
                "similarity": mat_c_sparse.data,
            }
        )
        self.similarity = State().session.createDataFrame(mat_c_pd).cache()

    def _main_iteration(self, inv_matrix, p_x):
        self._mat_b = self._calc_b(
            inv_matrix, p_x, self._mat_c, self._mat_gamma
        )
        prev_mat_c = self._mat_c
        self._mat_c = self._calc_c(self._mat_b, self._mat_gamma)
        self._mat_gamma += self.rho * (self._mat_b - self._mat_c)
        r_primal = np.linalg.norm(self._mat_b - self._mat_c)
        r_dual = np.linalg.norm(-self.rho * (self._mat_c - prev_mat_c))
        eps_primal = self.eps_abs * self.items_count + self.eps_rel * max(
            np.linalg.norm(self._mat_b), np.linalg.norm(self._mat_c)
        )
        eps_dual = (
            self.eps_abs * self.items_count
            + self.eps_rel * np.linalg.norm(self._mat_gamma)
        )
        if r_primal > self.threshold * r_dual:
            self.rho *= self.multiplicator
        elif self.threshold * r_primal < r_dual:
            self.rho /= self.multiplicator

        return r_primal, r_dual, eps_primal, eps_dual

    def _init_matrix(
        self, size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Начальная инициализвция матриц"""
        np.random.seed(self.seed)
        mat_b = np.random.rand(size, size)
        mat_c = np.random.rand(size, size)
        mat_gamma = np.random.rand(size, size)
        return mat_b, mat_c, mat_gamma

    def _calc_b(
        self,
        inv_matrix: np.ndarray,
        p_x: np.ndarray,
        mat_c: np.ndarray,
        mat_gamma: np.ndarray,
    ) -> np.ndarray:
        """Вычисление матрицы B"""
        mat_b = p_x + (inv_matrix @ (self.rho * mat_c - mat_gamma))
        vec_gamma = np.diag(mat_b) / np.diag(inv_matrix)
        return mat_b - inv_matrix * vec_gamma

    def _calc_c(self, mat_b: np.ndarray, mat_gamma: np.ndarray) -> np.ndarray:
        """Вычисление матрицы C"""
        mat_c = mat_b + mat_gamma / self.rho
        coef = self.lambda_1 / self.rho
        s_k = np.maximum(mat_c - coef, 0.0) - np.maximum(-mat_c - coef, 0.0)
        return s_k

    # pylint: disable=too-many-arguments
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
        recs = (
            users.withColumnRenamed("user_idx", "user")
            .join(
                log.withColumnRenamed("item_idx", "item"),
                how="inner",
                on=sf.col("user") == sf.col("user_idx"),
            )
            .join(
                self.similarity,
                how="inner",
                on=sf.col("item") == sf.col("item_id_one"),
            )
            .join(
                items,
                how="inner",
                on=sf.col("item_idx") == sf.col("item_id_two"),
            )
            .groupby("user_idx", "item_idx")
            .agg(sf.sum("similarity").alias("relevance"))
            .select("user_idx", "item_idx", "relevance")
            .cache()
        )

        return recs
