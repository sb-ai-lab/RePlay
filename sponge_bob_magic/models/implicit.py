from typing import Optional, Dict

from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from pyspark.sql import functions as sf

from sponge_bob_magic.models import Recommender


class ImplicitWrap(Recommender):
    """Обертка для пакета `Implicit
    <https://github.com/benfred/implicit>`_"""

    def __init__(self, model):
        """На вход принимаестя инициализированная модель Implicit."""
        self.model = model

    def get_params(self) -> Dict[str, object]:
        return {}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        log = self.item_indexer.transform(self.user_indexer.transform(log))
        matrix = to_csr(log)
        self.model.fit(matrix)
        self.user_item_data = matrix.T.tocsr()

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
        @sf.pandas_udf(
            "user_id int, item_idx int, relevance double",
            sf.PandasUDFType.GROUPED_MAP,
        )
        def predict_by_user(pandas_df):
            user = int(
                pandas_df["user_idx"].iloc[0]
            )  # сюда приходит одна строка
            user_id = pandas_df["user_id"].iloc[0]
            res = model.recommend(
                user, user_item_data, k, filter_seen_items, items_to_drop
            )
            return pd.DataFrame(
                {
                    "user_id": [user_id] * len(res),
                    "item_idx": [val[0] for val in res],
                    "relevance": [val[1] for val in res],
                }
            )

        items_to_drop = self._invert_items(log, items)
        user_item_data = self.user_item_data
        model = self.model
        recs = self.user_indexer.transform(users)
        recs = (
            recs.select("user_idx", "user_id")
            .groupby("user_idx")
            .apply(predict_by_user)
        )
        return self.inv_item_indexer.transform(recs).drop(
            "user_idx", "item_idx"
        )

    def _invert_items(self, log: DataFrame, items: DataFrame) -> list:
        """
        В функцию передаются айтемы, для которых нужно сделать предикт,
        а implicit ожидает айтемы, до которых не нужно делать предикт.

        Данная функция выделяет все айтемы из лога и убирает те, которые есть в items.
        Оставшееся -- то, что необходимо выкинуть.

        Кроме того, индексы переводятся во внутренний формат.
        """
        all_items = self._extract_unique(log, None, "item_id")
        res = all_items.select(all_items.item_id).where(
            ~all_items.item_id.isin(items.item_id)
        )
        return self.item_indexer.transform(res).toPandas().item_idx.to_list()


def to_csr(log: DataFrame) -> csr_matrix:
    """Конвертирует лог в csr матрицу."""
    users = to_numpy(log, "user_idx")
    items = to_numpy(log, "item_idx")
    relevance = to_numpy(log, "relevance")
    return csr_matrix((relevance, (items, users)))


def to_numpy(log: DataFrame, col: str) -> np.array:
    """Преобразует тип колонки спарк датафрейма"""
    return np.concatenate(
        log.select(col)
        .rdd.glom()
        .map(lambda x: np.array([elem[0] for elem in x]))
        .collect()
    ).astype(int)
