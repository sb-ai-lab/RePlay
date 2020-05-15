from typing import Optional, Dict

from pyspark.sql import DataFrame
import pandas as pd
from pyspark.sql import functions as sf

from sponge_bob_magic.models import Recommender
from sponge_bob_magic.utils import to_csr


class ImplicitWrap(Recommender):
    """Обертка для пакета `implicit
    <https://github.com/benfred/implicit>`_

    Пример:

    >>> import implicit
    >>> model = implicit.als.AlternatingLeastSquares(factors=5)
    >>> als = ImplicitWrap(model)

    Теперь модель можно использовать как любую другую в библиотеке.
    Обертка обеспеивает конвертацию датафреймов в матрицы и единый интерфейс.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2, 2], "item_id": [1, 2, 2, 3], "relevance": [1, 1, 1, 1]})
    >>> als.fit_predict(df, 1, users=[1])[["user_id", "item_id"]]
      user_id item_id
    0       1       3

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(item_user_data)

    # recommend items for a user
    user_items = item_user_data.T.tocsr()
    recommendations = model.recommend(userid, user_items)

    # find related items
    related = model.similar_items(itemid)

    """

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
        matrix = to_csr(log)
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
        user_item_data = to_csr(log).T.tocsr()
        model = self.model
        return (
            users.select("user_idx").groupby("user_idx").apply(predict_by_user)
        )

    def _invert_items(self, log: DataFrame, items: DataFrame) -> list:
        """
        В функцию передаются айтемы, для которых нужно сделать предикт,
        а implicit ожидает айтемы, до которых не нужно делать предикт.

        Данная функция выделяет все айтемы из лога и убирает те, которые есть в items.
        Оставшееся -- то, что необходимо выкинуть.

        Кроме того, индексы переводятся во внутренний формат.
        """
        return (  # type: ignore
            log.select("item_idx")
            .subtract(items)
            .select("item_idx")
            .toPandas()
            .item_idx.to_list()
        )
