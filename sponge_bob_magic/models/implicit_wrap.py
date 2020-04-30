from typing import Optional, Dict

from pyspark.sql import DataFrame
import pandas as pd
from pyspark.sql import functions as sf

from sponge_bob_magic.models import Recommender
from sponge_bob_magic.utils import to_csr


class ImplicitWrap(Recommender):  # pragma: no cover
    """Обертка для пакета `implicit
    <https://github.com/benfred/implicit>`_"""

    def __init__(self, model):
        """На вход принимаестя инициализированная модель implicit."""
        self.model = model

    def get_params(self) -> Dict[str, object]:
        return {}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        matrix = to_csr(self.index(log))
        self.model.fit(matrix)

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
            "user_idx int, item_idx int, relevance double",
            sf.PandasUDFType.GROUPED_MAP,
        )
        def predict_by_user(pandas_df):
            user = int(pandas_df["user_idx"].iloc[0])
            res = model.recommend(
                user, user_item_data, k, filter_seen_items, items_to_drop
            )
            return pd.DataFrame(
                {
                    "user_idx": [user] * len(res),
                    "item_idx": [val[0] for val in res],
                    "relevance": [val[1] for val in res],
                }
            )

        items_to_drop = self._invert_items(log, items)
        user_item_data = to_csr(self.index(log)).T.tocsr()
        model = self.model
        recs = self.user_indexer.transform(users)
        recs = (
            recs.select("user_idx").groupby("user_idx").apply(predict_by_user)
        )
        return self.inv_index(recs)

    def _invert_items(self, log: DataFrame, items: DataFrame) -> list:
        """
        В функцию передаются айтемы, для которых нужно сделать предикт,
        а implicit ожидает айтемы, до которых не нужно делать предикт.

        Данная функция выделяет все айтемы из лога и убирает те, которые есть в items.
        Оставшееся -- то, что необходимо выкинуть.

        Кроме того, индексы переводятся во внутренний формат.
        """
        all_items = self._extract_unique(log, None, "item_id")
        return (
            self.item_indexer.transform(
                all_items.select("item_id").subtract(items)
            )
            .select("item_idx")
            .toPandas()
            .item_idx.to_list()
        )
