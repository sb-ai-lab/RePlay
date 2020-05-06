"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Optional

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from statsmodels.stats.proportion import proportion_confint

from sponge_bob_magic.converter import convert
from sponge_bob_magic.models.pop_rec import PopRec


class Wilson(PopRec):
    """
    Подсчитевает для каждого айтема нижнюю границу
    доверительного интервала истинной доли положительных оценок.

    .. math::
        WilsonScore = \\frac{\\widehat{p}+\\frac{z_{ \\frac{\\alpha}{2}}^{2}}{2n}\\pm z_
        {\\frac{\\alpha}{2}}\\sqrt{\\frac{\\widehat{p}(1-\\widehat{p})+\\frac{z_
        {\\frac{\\alpha}{2}}^{2}}{4n}}{n}} }{1+\\frac{z_{ \\frac{\\alpha}{2}}^{2}}{n}}


    Где :math:`\hat{p}` -- наблюдаемая доля положительных оценок (1 по отношению к 0).

    :math:`z_{\\alpha}` 1-альфа квантиль нормального распределения.

    Для каждого пользователя отфильтровываются просмотренные айтемы.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2], "relevance": [1, 1]})
    >>> model = Wilson()
    >>> model.fit_predict(data_frame, k=1)
       user_id item_id  relevance
    0        1       2   0.206549
    1        2       1   0.206549

    """

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        log = convert(log)

        data_frame = (
            log.groupby("item_id")
            .agg(
                sf.sum("relevance").alias("pos"),
                sf.count("relevance").alias("total"),
            )
            .toPandas()
        )
        pos = np.array(data_frame.pos)
        total = np.array(data_frame.total)
        data_frame["relevance"] = proportion_confint(
            pos, total, method="wilson"
        )[0]
        data_frame = data_frame.drop(["pos", "total"], axis=1)
        self.item_popularity = convert(data_frame).cache()
