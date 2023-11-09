from typing import Optional

from replay.data.dataset import DataFrameLike, Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.utils.spark_utils import convert2spark


# pylint: disable=too-many-arguments
def create_dataset(
    interactions: DataFrameLike,
    query_features: Optional[DataFrameLike] = None,
    item_features: Optional[DataFrameLike] = None,
    feature_schema: Optional[DataFrameLike] = None,
    query_column: str = "user_id",
    item_column: str = "item_id",
    rating_column: str = "rating",
    timestamp_column: str = "timestamp",
    has_rating: bool = True,
    has_timestamp: bool = False,
) -> Dataset:
    """
    Get Dataset instance with given dataframes. Converts data to Spark DataFrames

    :param interactions: historical interactions
    :param query_features: user/query features
    :param item_features: item features
    :param feature_schema: instance of FeatureSchema
    :param query_column: name of column with query ids
    :param item_column: name of column with item ids
    :param rating_column: name of column with ratings
    :param timestamp_column: name of column with timestamp
    :param has_rating: flag for existing rating column in interactions
    :param has_timestamp: flag for existing timestamp column in interactions
    """
    interactions = convert2spark(interactions)
    if query_features is not None:
        query_features = convert2spark(query_features)
    if item_features is not None:
        item_features = convert2spark(item_features)

    if feature_schema is None:
        base = [
            FeatureInfo(
                column=query_column,
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.QUERY_ID,
            ),
            FeatureInfo(
                column=item_column,
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
        ]
        rating_feature = [
            FeatureInfo(
                column=rating_column,
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.RATING,
            ),
        ]
        time_feature = [
            FeatureInfo(
                column=timestamp_column,
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]

        feature_infos = base
        if has_rating:
            feature_infos += rating_feature
        if has_timestamp:
            feature_infos += time_feature

        feature_schema = FeatureSchema(feature_infos)
    return Dataset(
        feature_schema=feature_schema,
        interactions=interactions,
        query_features=query_features,
        item_features=item_features,
        check_consistency=True,
    )
