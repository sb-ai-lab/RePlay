from replay.utils.common import convert2pandas, convert2polars, convert2spark

from .base_rec_client import NonPersonolizedRecommenderClient
from .implementations import _PopRecPandas, _PopRecPolars, _PopRecSpark


class PopRec(NonPersonolizedRecommenderClient):
    """
    Recommend objects using their popularity.

    Popularity of an item is a probability that random user rated this item.

    .. math::
        Popularity(i) = \\dfrac{N_i}{N}

    :math:`N_i` - number of users who rated item :math:`i`

    :math:`N` - total number of users

    >>> import pandas as pd
    >>> from replay.data.dataset import Dataset, FeatureSchema, FeatureInfo, FeatureHint, FeatureType
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = pd.DataFrame(
    ...    {"user_id": [1, 1, 2, 2, 3, 4],
    ...     "item_id": [1, 2, 2, 3, 3, 3],
    ...     "rating": [0.5, 1, 0.1, 0.8, 0.7, 1]}
    ... )
    >>> data_frame
        user_id   item_id     rating
    0         1         1        0.5
    1         1         2        1.0
    2         2         2        0.1
    3         2         3        0.8
    4         3         3        0.7
    5         4         3        1.0

    >>> feature_schema = FeatureSchema(
    ...     [
    ...         FeatureInfo(
    ...             column="user_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.QUERY_ID,
    ...         ),
    ...         FeatureInfo(
    ...             column="item_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.ITEM_ID,
    ...         ),
    ...         FeatureInfo(
    ...             column="rating",
    ...             feature_type=FeatureType.NUMERICAL,
    ...             feature_hint=FeatureHint.RATING,
    ...         ),
    ...     ]
    ... )
    >>> interactions = convert2spark(data_frame)
    >>> dataset = Dataset(feature_schema, interactions)
    >>> res = PopRec().fit_predict(dataset, 1)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
        user_id   item_id     rating
    0         1         3       0.75
    1         2         1       0.25
    2         3         2       0.50
    3         4         2       0.50

    >>> res = PopRec().fit_predict(dataset, 1, filter_seen_items=False)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
        user_id   item_id     rating
    0         1         3       0.75
    1         2         3       0.75
    2         3         3       0.75
    3         4         3       0.75

    >>> res = PopRec(use_rating=True).fit_predict(dataset, 1)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
        user_id   item_id     rating
    0         1         3      0.625
    1         2         1      0.125
    2         3         2      0.275
    3         4         2      0.275

    """

    _class_map = {"spark": _PopRecSpark, "pandas": _PopRecPandas, "polars": _PopRecPolars}

    def __init__(
        self,
        use_rating: bool = False,
        add_cold_items: bool = True,
        cold_weight: float = 0.5,
    ):
        """
        :param use_rating: flag to use rating values as is or to treat them as 1
        :param add_cold_items: flag to consider cold items in recommendations building
            if present in `items` parameter of `predict` method
            or `pairs` parameter of `predict_pairs` methods.
            If true, cold items are assigned rating equals to the less relevant item rating
            multiplied by cold_weight and may appear among top-K recommendations.
            Otherwise cold items are filtered out.
            Could be changed after model training by setting the `add_cold_items` attribute.
        : param cold_weight: if `add_cold_items` is True,
            cold items are added with reduced rating.
            The rating for cold items is equal to the rating
            of a least relevant item multiplied by a `cold_weight` value.
            `Cold_weight` value should be in interval (0, 1].
        """
        self._use_rating = use_rating
        super().__init__(add_cold_items=add_cold_items, cold_weight=cold_weight)

    @property
    def use_rating(self):
        if self.is_fitted:
            return self._impl.use_rating
        else:
            return self._use_rating

    @use_rating.setter
    def use_rating(self, value: bool):
        if not isinstance(value, bool):
            msg = f"incorrect type of argument 'value' ({type(value)}). Use bool"
            raise ValueError(msg)
        self._use_rating = value
        if self.is_fitted:
            self._impl.use_rating = self._use_rating
        else:
            self._init_when_first_impl_arrived_args.update({"use_rating": value})

    @property
    def _init_args(self):
        if not hasattr(self._impl, "_init_args"):
            return {
                "use_rating": self._use_rating,
                "add_cold_items": self._add_cold_items,
                "cold_weight": self._cold_weight,
            }
        return self._impl._init_args

    def to_pandas(self):
        if self.is_fitted:
            item_popularity = convert2pandas(self.item_popularity)
            fill = self.fill
        same_object = super().to_pandas()
        if self.is_fitted:
            same_object.item_popularity = item_popularity
            same_object.fill = fill
        return same_object

    def to_spark(self):
        if self.is_fitted:
            item_popularity = convert2spark(self.item_popularity)
            fill = self.fill
        same_object = super().to_spark()
        if self.is_fitted:
            same_object.item_popularity = item_popularity
            same_object.fill = fill
        return same_object

    def to_polars(self):
        if self.is_fitted:
            item_popularity = convert2polars(self.item_popularity)
            fill = self.fill
        copy_realization = super().to_polars()
        if self.is_fitted:
            copy_realization.item_popularity = item_popularity
            copy_realization.fill = fill
        return self
