from typing import Optional

from pyspark.sql import functions as sf
from pyspark.sql import DataFrame
import numpy as np
import scipy.stats as st

from sponge_bob_magic.converter import convert
from sponge_bob_magic.models import PopRec


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
    """

    def _pre_fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        super(PopRec, self)._pre_fit(log, None, None)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        log = convert(log)

        pos = log.groupby("item_id").agg(
            sf.sum((sf.col("relevance") == 1).astype("int")).alias("pos")
        )
        neg = log.groupby("item_id").agg(
            sf.sum((sf.col("relevance") == 0).astype("int")).alias("neg")
        )
        df = pos.join(neg, on="item_id").toPandas()

        pos = np.array(df.pos)
        neg = np.array(df.neg)

        df["relevance"] = wilson_score(pos, neg)
        df = df.drop(["pos", "neg"], axis=1)
        self.item_popularity = convert(df).cache()


def wilson_score(ups: np.array, downs: np.array, confidence: float = 0.85):
    """
    Рассчитывает wilson score для массивов с количеством лайков и дизлайков по айтемам.
    :param ups: количество лайков
    :param downs: количество дизлайков
    :param confidence: доверительный интервал
    :return: массив рассчитанных значений
    """
    n = ups + downs
    phat = ups / n
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    a = phat + z * z / (2 * n)
    b = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    c = 1 + z * z / n
    return (a - b) / c
