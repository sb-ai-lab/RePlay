"""
``Dataset`` universal dataset class for manipulating interactions and feed data to models.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame

from .schema import FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.storagelevel import StorageLevel


class Dataset:
    """
    Universal dataset for feeding data to models.
    """

    def __init__(
        self,
        feature_schema: FeatureSchema,
        interactions: DataFrameLike,
        query_features: Optional[DataFrameLike] = None,
        item_features: Optional[DataFrameLike] = None,
        check_consistency: bool = True,
        categorical_encoded: bool = False,
    ):
        """
        :param feature_schema: mapping of columns names and feature infos.
        :param interactions: dataframe with interactions.
        :param query_features: dataframe with query features,
            defaults: ```None```.
        :param item_features: dataframe with item features,
            defaults: ```None```.
        :param check_consistency: the parameter responsible for checking the consistency of the data,
            defaults: ```True```.
        :param categorical_encoded: the parameter responsible for checking the categorical features
            encoded validity,
            defaults: ```False```.
        """
        self._interactions = interactions
        self._query_features = query_features
        self._item_features = item_features

        self.is_pandas = isinstance(interactions, PandasDataFrame)
        self.is_spark = isinstance(interactions, SparkDataFrame)
        self.is_polars = isinstance(interactions, PolarsDataFrame)

        self._categorical_encoded = categorical_encoded

        try:
            feature_schema.item_id_column
        except Exception as exception:
            msg = "Item id column is not set."
            raise ValueError(msg) from exception

        try:
            feature_schema.query_id_column
        except Exception as exception:
            msg = "Query id column is not set."
            raise ValueError(msg) from exception

        if self.item_features is not None and not check_dataframes_types_equal(self._interactions, self.item_features):
            msg = "Interactions and item features should have the same type."
            raise TypeError(msg)
        if self.query_features is not None and not check_dataframes_types_equal(
            self._interactions, self.query_features
        ):
            msg = "Interactions and query features should have the same type."
            raise TypeError(msg)

        self._feature_source_map: Dict[FeatureSource, DataFrameLike] = {
            FeatureSource.INTERACTIONS: self.interactions,
            FeatureSource.QUERY_FEATURES: self.query_features,
            FeatureSource.ITEM_FEATURES: self.item_features,
        }

        self._ids_feature_map: Dict[FeatureHint, DataFrameLike] = {
            FeatureHint.QUERY_ID: self.query_features if self.query_features is not None else self.interactions,
            FeatureHint.ITEM_ID: self.item_features if self.item_features is not None else self.interactions,
        }

        self._feature_schema = self._fill_feature_schema(feature_schema)

        if check_consistency:
            if self.query_features is not None:
                self._check_ids_consistency(hint=FeatureHint.QUERY_ID)
            if self.item_features is not None:
                self._check_ids_consistency(hint=FeatureHint.ITEM_ID)

            if self._categorical_encoded:
                self._check_encoded()

    @property
    def is_categorical_encoded(self) -> bool:
        """
        :returns: is categorical features are encoded.
        """
        return self._categorical_encoded

    @property
    def interactions(self) -> DataFrameLike:
        """
        :returns: interactions dataset.
        """
        return self._interactions

    @property
    def query_features(self) -> Optional[DataFrameLike]:
        """
        :returns: query features dataset.
        """
        return self._query_features

    @property
    def item_features(self) -> Optional[DataFrameLike]:
        """
        :returns: item features dataset.
        """
        return self._item_features

    @property
    def query_ids(self) -> DataFrameLike:
        """
        :returns: dataset with unique query ids.
        """
        query_column_df = self._ids_feature_map[FeatureHint.QUERY_ID]
        if self.is_pandas:
            assert isinstance(query_column_df, PandasDataFrame)
            query_ids = PandasDataFrame(
                {self.feature_schema.query_id_column: query_column_df[self.feature_schema.query_id_column].unique()}
            )
        if self.is_spark:
            assert isinstance(query_column_df, SparkDataFrame)
            query_ids = query_column_df.select(self.feature_schema.query_id_column).distinct()
        if self.is_polars:
            assert isinstance(query_column_df, PolarsDataFrame)
            query_ids = query_column_df.select(self.feature_schema.query_id_column).unique()

        return query_ids

    @property
    def item_ids(self) -> DataFrameLike:
        """
        :returns: dataset with unique item ids.
        """
        item_column_df = self._ids_feature_map[FeatureHint.ITEM_ID]
        if self.is_pandas:
            assert isinstance(item_column_df, PandasDataFrame)
            all_item_ids = item_column_df[self.feature_schema.item_id_column]
            unique_item_ids = all_item_ids.unique()
            item_ids = PandasDataFrame({self.feature_schema.item_id_column: unique_item_ids})
        if self.is_spark:
            assert isinstance(item_column_df, SparkDataFrame)
            item_ids = item_column_df.select(self.feature_schema.item_id_column).distinct()
        if self.is_polars:
            assert isinstance(item_column_df, PolarsDataFrame)
            item_ids = item_column_df.select(self.feature_schema.item_id_column).unique()

        return item_ids

    @property
    def query_count(self) -> int:
        """
        :returns: the number of queries.
        """
        query_count = self.feature_schema.query_id_feature.cardinality
        assert query_count is not None
        return query_count

    @property
    def item_count(self) -> int:
        """
        :returns: The number of items.
        """
        item_count = self.feature_schema.item_id_feature.cardinality
        assert item_count is not None
        return item_count

    @property
    def feature_schema(self) -> FeatureSchema:
        """
        :returns: List of features.
        """
        return self._feature_schema

    if PYSPARK_AVAILABLE:

        def persist(self, storage_level: StorageLevel = StorageLevel(True, True, False, True, 1)) -> None:
            """
            Sets the storage level to persist SparkDataFrame for interactions, item_features
            and user_features.

            The function is only available when the PySpark is installed.

            :param storage_level: storage level to set for persistance.
                default: ```MEMORY_AND_DISK_DESER```.
            """
            if self.is_spark:
                self.interactions.persist(storage_level)
                if self.item_features is not None:
                    self.item_features.persist(storage_level)
                if self.query_features is not None:
                    self.query_features.persist(storage_level)

        def unpersist(self, blocking: bool = False) -> None:
            """
            Marks SparkDataFrame as non-persistent, and remove all blocks for it from memory and disk
            for interactions, item_features and user_features.

            The function is only available when the PySpark is installed.

            :param blocking: whether to block until all blocks are deleted.
                default: ```False```.
            """
            if self.is_spark:
                self.interactions.unpersist(blocking)
                if self.item_features is not None:
                    self.item_features.unpersist(blocking)
                if self.query_features is not None:
                    self.query_features.unpersist(blocking)

        def cache(self) -> None:
            """
            Persists the SparkDataFrame with the default storage level (MEMORY_AND_DISK)
            for interactions, item_features and user_features.

            The function is only available when the PySpark is installed.
            """
            if self.is_spark:
                self.interactions.cache()
                if self.item_features is not None:
                    self.item_features.cache()
                if self.query_features is not None:
                    self.query_features.cache()

    def subset(self, features_to_keep: Iterable[str]) -> Dataset:
        """
        Returns subset of features. Keeps query and item IDs even if
        the corresponding sources are not explicitly passed to this functions.

        :param features_to_keep: sequence of features to keep.

        :returns: new Dataset with given features.
        """
        # We always need to have query and item ID features in interactions dataset
        features_to_keep_set = set(features_to_keep)
        features_to_keep_set.add(self._feature_schema.query_id_column)
        features_to_keep_set.add(self._feature_schema.item_id_column)

        feature_schema_subset = self._feature_schema.subset(features_to_keep_set)

        interaction_fi = (
            FeatureSchema(feature_schema_subset.query_id_feature)
            + FeatureSchema(feature_schema_subset.item_id_feature)
            + feature_schema_subset.interaction_features
        )
        interactions = select(self._interactions, interaction_fi.columns)

        query_features = self._query_features
        if query_features is not None:
            query_fi = FeatureSchema(feature_schema_subset.query_id_feature) + feature_schema_subset.query_features
            query_features = select(query_features, query_fi.columns)

        item_features = self._item_features
        if item_features is not None:
            item_fi = FeatureSchema(feature_schema_subset.item_id_feature) + feature_schema_subset.item_features
            item_features = select(item_features, item_fi.columns)

        # We do not need to check consistency as it was already checked during parent dataset creation
        # Taking subset does not modify column values
        return Dataset(
            feature_schema=feature_schema_subset,
            interactions=interactions,
            query_features=query_features,
            item_features=item_features,
            check_consistency=False,
            categorical_encoded=self._categorical_encoded,
        )

    def _get_cardinality(self, feature: FeatureInfo) -> Callable:
        def callback(column: str) -> int:
            if feature.feature_hint in [FeatureHint.ITEM_ID, FeatureHint.QUERY_ID]:
                return nunique(self._ids_feature_map[feature.feature_hint], column)
            assert feature.feature_source
            return nunique(self._feature_source_map[feature.feature_source], column)

        return callback

    def _set_cardinality(self, features_list: Sequence[FeatureInfo]) -> None:
        for feature in features_list:
            if feature.feature_type == FeatureType.CATEGORICAL:
                feature._set_cardinality_callback(self._get_cardinality(feature))

    def _fill_feature_schema(self, feature_schema: FeatureSchema) -> FeatureSchema:
        features_list = self._fill_unlabeled_features_sources(feature_schema=feature_schema)
        updated_feature_schema = FeatureSchema(features_list)

        filled_features = self._fill_unlabeled_features(
            source=FeatureSource.INTERACTIONS,
            feature_schema=updated_feature_schema,
        )

        if self.item_features is not None:
            filled_features += self._fill_unlabeled_features(
                source=FeatureSource.ITEM_FEATURES,
                feature_schema=updated_feature_schema,
            )

        if self.query_features is not None:
            filled_features += self._fill_unlabeled_features(
                source=FeatureSource.QUERY_FEATURES,
                feature_schema=updated_feature_schema,
            )
        return FeatureSchema(features_list=features_list + filled_features)

    def _fill_unlabeled_features_sources(self, feature_schema: FeatureSchema) -> List[FeatureInfo]:
        features_list = list(feature_schema.all_features)

        source_mapping: Dict[str, FeatureSource] = {}
        for source in FeatureSource:
            dataframe = self._feature_source_map[source]
            if dataframe is not None:
                for column in dataframe.columns:
                    if column in feature_schema.columns:
                        source_mapping[column] = source

        for feature in features_list:
            if feature.feature_hint in [FeatureHint.QUERY_ID, FeatureHint.ITEM_ID]:
                feature._set_feature_source(source=FeatureSource.INTERACTIONS)
                continue
            source = source_mapping.get(feature.column)
            if source:
                feature._set_feature_source(source=source_mapping[feature.column])
            else:
                msg = f"{feature.column} doesn't exist in provided dataframes"
                raise ValueError(msg)

        self._set_cardinality(features_list=features_list)
        return features_list

    def _get_unlabeled_columns(self, source: FeatureSource, feature_schema: FeatureSchema) -> List[FeatureInfo]:
        set_source_dataframe_columns = set(self._feature_source_map[source].columns)
        set_labeled_dataframe_columns = set(feature_schema.columns)
        unlabeled_columns = set_source_dataframe_columns - set_labeled_dataframe_columns
        unlabeled_features_list = [
            FeatureInfo(column=column, feature_source=source, feature_type=FeatureType.NUMERICAL)
            for column in unlabeled_columns
        ]
        return unlabeled_features_list

    def _fill_unlabeled_features(self, source: FeatureSource, feature_schema: FeatureSchema) -> List[FeatureInfo]:
        unlabeled_columns = self._get_unlabeled_columns(source=source, feature_schema=feature_schema)
        self._set_features_source(feature_list=unlabeled_columns, source=source)
        self._set_cardinality(features_list=unlabeled_columns)
        return unlabeled_columns

    def _set_features_source(self, feature_list: List[FeatureInfo], source: FeatureSource) -> None:
        for feature in feature_list:
            feature._set_feature_source(source)

    def _check_ids_consistency(self, hint: FeatureHint) -> None:
        """
        Checks that all the ids from the interactions are in the features dataframe.
        """
        features_df = self._ids_feature_map[hint]
        ids_column = (
            self.feature_schema.item_id_column if hint == FeatureHint.ITEM_ID else self.feature_schema.query_id_column
        )
        if self.is_pandas:
            interactions_unique_ids = set(self.interactions[ids_column].unique())
            features_df_unique_ids = set(features_df[ids_column].unique())
            in_interactions_not_in_features_ids = interactions_unique_ids - features_df_unique_ids
            is_consistent = len(in_interactions_not_in_features_ids) == 0
        elif self.is_spark:
            is_consistent = (
                self.interactions.select(ids_column)
                .distinct()
                .join(features_df.select(ids_column).distinct(), on=[ids_column], how="leftanti")
                .count()
            ) == 0
        else:
            is_consistent = (
                len(
                    self.interactions.select(ids_column)
                    .unique()
                    .join(features_df.select(ids_column).unique(), on=ids_column, how="anti")
                )
                == 0
            )

        if not is_consistent:
            msg = f"There are IDs in the interactions that are missing in the {hint.name} dataframe."
            raise ValueError(msg)

    def _check_column_encoded(
        self, data: DataFrameLike, column: str, source: FeatureSource, cardinality: Optional[int]
    ) -> None:
        """
        Checks that IDs are encoded:
        1) IDs are integers;
        2) Min id >= 0;
        3) Max id < quantity of unique IDs.

        TODO: think about the third criterion. Case when full data was encoded and then splitted.
        Option: Keep this criterion, but suggest the user to disable the check if he understands
        that the criterion will not pass.
        """
        if self.is_pandas:
            is_int = np.issubdtype(dict(data.dtypes)[column], int)
        elif self.is_spark:
            is_int = "int" in dict(data.dtypes)[column]
        else:
            is_int = data[column].dtype.is_integer()

        if not is_int:
            msg = f"IDs in {source.name}.{column} are not encoded. They are not int."
            raise ValueError(msg)

        if self.is_pandas:
            min_id = data[column].min()
        elif self.is_spark:
            min_id = data.agg(sf.min(column).alias("min_index")).collect()[0][0]
        else:
            min_id = data[column].min()
        if min_id < 0:
            msg = f"IDs in {source.name}.{column} are not encoded. Min ID is less than 0."
            raise ValueError(msg)

        if self.is_pandas:
            max_id = data[column].max()
        elif self.is_spark:
            max_id = data.agg(sf.max(column).alias("max_index")).collect()[0][0]
        else:
            max_id = data[column].max()

        if max_id >= cardinality:
            msg = f"IDs in {source.name}.{column} are not encoded. Max ID is more than quantity of IDs."
            raise ValueError(msg)

    def _check_encoded(self) -> None:
        for feature in self.feature_schema.categorical_features.all_features:
            if feature.feature_hint == FeatureHint.ITEM_ID:
                self._check_column_encoded(
                    self.interactions,
                    feature.column,
                    FeatureSource.INTERACTIONS,
                    feature.cardinality,
                )
                if self.item_features is not None:
                    self._check_column_encoded(
                        self.item_features,
                        feature.column,
                        FeatureSource.ITEM_FEATURES,
                        feature.cardinality,
                    )
            elif feature.feature_hint == FeatureHint.QUERY_ID:
                self._check_column_encoded(
                    self.interactions,
                    feature.column,
                    FeatureSource.INTERACTIONS,
                    feature.cardinality,
                )
                if self.query_features is not None:
                    self._check_column_encoded(
                        self.query_features,
                        feature.column,
                        FeatureSource.QUERY_FEATURES,
                        feature.cardinality,
                    )
            else:
                data = self._feature_source_map[feature.feature_source]
                self._check_column_encoded(
                    data,
                    feature.column,
                    feature.feature_source,
                    feature.cardinality,
                )


def nunique(data: DataFrameLike, column: str) -> int:
    """
    Returns number of unique values of specified column in dataframe.

    :param data: dataframe.
    :param column: column name.

    :returns: number of unique values.
    """
    if isinstance(data, SparkDataFrame):
        return data.select(column).distinct().count()
    if isinstance(data, PandasDataFrame):
        return data[column].nunique()
    return data.select(column).n_unique()


def select(data: DataFrameLike, columns: Sequence[str]) -> DataFrameLike:
    """
    :param data: dataframe.
    :param columns: sequence of column names to select.

    :returns: selected data in the same format as input dataframe.
    """
    if isinstance(data, SparkDataFrame):
        return data.select(*columns)
    if isinstance(data, PandasDataFrame):
        return data[columns]
    if isinstance(data, PolarsDataFrame):
        return data.select(*columns)
    assert False, "Unknown data frame type"


def check_dataframes_types_equal(dataframe: DataFrameLike, other: DataFrameLike):
    """
    :param dataframe: dataframe.
    :param other: dataframe to compare.

    :returns: True if dataframes have same type.
    """
    if isinstance(dataframe, PandasDataFrame) and isinstance(other, PandasDataFrame):
        return True
    if isinstance(dataframe, SparkDataFrame) and isinstance(other, SparkDataFrame):
        return True
    if isinstance(dataframe, PolarsDataFrame) and isinstance(other, PolarsDataFrame):
        return True
    return False
