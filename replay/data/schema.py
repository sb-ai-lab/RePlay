from enum import Enum
from typing import (
    Callable,
    Dict,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
    ValuesView,
)


class FeatureType(Enum):
    """Type of Feature."""

    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


class FeatureSource(Enum):
    """Name of DataFrame."""

    ITEM_FEATURES = "item_features"
    QUERY_FEATURES = "query_features"
    INTERACTIONS = "interactions"


class FeatureHint(Enum):
    """Hint to algorithm about column."""

    ITEM_ID = "item_id"
    QUERY_ID = "query_id"
    RATING = "rating"
    TIMESTAMP = "timestamp"


class FeatureInfo:
    """
    Information about a feature.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        column: str,
        feature_type: FeatureType,
        feature_hint: Optional[FeatureHint] = None,
        feature_source: Optional[FeatureSource] = None,
        cardinality: Optional[int] = None,
    ) -> None:
        """
        :param column: name of feature.
        :param feature_type: type of feature.
        :param feature_hint: hint to models about feature
            (is timestamp, is rating, is query_id, is item_id),
            default: ``None``.
        :param feature_source: name of DataFrame feature came from,
            default: ``None``.
        :param cardinality: cardinality of categorical feature, required for ids columns,
            optional for others,
            default: ``None``.
        """
        self._column = column
        self._feature_type = feature_type
        self._feature_source = feature_source
        self._feature_hint = feature_hint

        if feature_type == FeatureType.NUMERICAL and cardinality:
            raise ValueError("Cardinality is needed only with categorical feature_type.")
        self._cardinality = cardinality

    @property
    def column(self) -> str:
        """
        :returns: the feature name.
        """
        return self._column

    @property
    def feature_type(self) -> FeatureType:
        """
        :returns: the type of feature.
        """
        return self._feature_type

    @property
    def feature_hint(self) -> Optional[FeatureHint]:
        """
        :returns: the feature hint.
        """
        return self._feature_hint

    @property
    def feature_source(self) -> Optional[FeatureSource]:
        """
        :returns: the name of source dataframe of feature.
        """
        return self._feature_source

    def _set_feature_source(self, source: FeatureSource) -> None:
        self._feature_source = source

    @property
    def cardinality(self) -> Optional[int]:
        """
        :returns: cardinality of the feature.
        """
        if self.feature_type != FeatureType.CATEGORICAL:
            raise RuntimeError(
                f"Can not get cardinality because feature_type of {self.column} column is not categorical."
            )
        if hasattr(self, "_cardinality_callback") and self._cardinality is None:
            self._cardinality = self._cardinality_callback(self._column)
        return self._cardinality

    # pylint: disable=attribute-defined-outside-init
    def _set_cardinality_callback(self, callback: Callable) -> None:
        self._cardinality_callback = callback

    def reset_cardinality(self) -> None:
        """
        Reset cardinality of the feature to None.
        """
        self._cardinality = None


# pylint: disable=too-many-public-methods
class FeatureSchema(Mapping[str, FeatureInfo]):
    """
    Key-value like collection with information about all dataset features.
    """

    def __init__(self, features_list: Union[Sequence[FeatureInfo], FeatureInfo]) -> None:
        """
        :param features_list: list of feature infos.
        """
        features_list = [features_list] if not isinstance(features_list, Sequence) else features_list
        self._check_features_naming(features_list)
        self._features_schema = {feature.column: feature for feature in features_list}

    def copy(self) -> "FeatureSchema":
        """
        Creates a copy of all features.

        :returns: copy of the initial feature schema.
        """
        copy_features_list = list(self._features_schema.values())
        for feature in copy_features_list:
            feature.reset_cardinality()
        return FeatureSchema(copy_features_list)

    def subset(self, features_to_keep: Iterable[str]) -> "FeatureSchema":
        """
        Creates a subset of given features.

        :param features_to_keep: a sequence of feature columns
            in original schema to keep in subset.
        :returns: new feature schema of given features.
        """
        features: Set[FeatureInfo] = set()
        for feature_column in features_to_keep:
            if feature_column in self._features_schema:
                features.add(self._features_schema[feature_column])
        return FeatureSchema(list(features))

    def item(self) -> FeatureInfo:
        """
        :returns: extract a feature information from a schema.
        """
        if len(self._features_schema) > 1:
            raise ValueError("Only one element feature schema can be converted to single feature")
        return list(self._features_schema.values())[0]

    def items(self) -> ItemsView[str, FeatureInfo]:
        return self._features_schema.items()

    def keys(self) -> KeysView[str]:
        return self._features_schema.keys()

    def values(self) -> ValuesView[FeatureInfo]:
        return self._features_schema.values()

    def get(  # type: ignore
        self,
        key: str,
        default: Optional[FeatureInfo] = None,
    ) -> Optional[FeatureInfo]:
        return self._features_schema.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._features_schema)

    def __contains__(self, feature_name: object) -> bool:
        return feature_name in self._features_schema

    def __len__(self) -> int:
        return len(self._features_schema)

    def __bool__(self) -> bool:
        return len(self._features_schema) > 0

    def __getitem__(self, feature_name: str) -> FeatureInfo:
        return self._features_schema[feature_name]

    def __eq__(self, other: object) -> bool:
        return self._features_schema == other

    def __ne__(self, other: object) -> bool:
        return self._features_schema != other

    def __add__(self, other: "FeatureSchema") -> "FeatureSchema":
        return FeatureSchema(list(self._features_schema.values()) + list(other._features_schema.values()))

    @property
    def all_features(self) -> Sequence[FeatureInfo]:
        """
        :returns: sequence of all features.
        """
        return list(self._features_schema.values())

    @property
    def categorical_features(self) -> "FeatureSchema":
        """
        :returns: sequence of categorical features in a schema.
        """
        return self.filter(feature_type=FeatureType.CATEGORICAL)

    @property
    def numerical_features(self) -> "FeatureSchema":
        """
        :returns: sequence of numerical features in a schema.
        """
        return self.filter(feature_type=FeatureType.NUMERICAL)

    @property
    def interaction_features(self) -> "FeatureSchema":
        """
        :returns: sequence of interaction features in a schema.
        """
        return (
            self.filter(feature_source=FeatureSource.INTERACTIONS)
            .drop(feature_hint=FeatureHint.ITEM_ID)
            .drop(feature_hint=FeatureHint.QUERY_ID)
        )

    @property
    def query_features(self) -> "FeatureSchema":
        """
        :returns: sequence of query features in a schema.
        """
        return self.filter(feature_source=FeatureSource.QUERY_FEATURES)

    @property
    def item_features(self) -> "FeatureSchema":
        """
        :returns: sequence of item features in a schema.
        """
        return self.filter(feature_source=FeatureSource.ITEM_FEATURES)

    @property
    def interactions_rating_features(self) -> "FeatureSchema":
        """
        :returns: sequence of interactions-rating features in a schema.
        """
        return self.filter(feature_source=FeatureSource.INTERACTIONS, feature_hint=FeatureHint.RATING)

    @property
    def interactions_timestamp_features(self) -> "FeatureSchema":
        """
        :returns: sequence of interactions-timestamp features in a schema.
        """
        return self.filter(feature_source=FeatureSource.INTERACTIONS, feature_hint=FeatureHint.TIMESTAMP)

    @property
    def columns(self) -> Sequence[str]:
        """
        :returns: list of all feature's column names.
        """
        return list(self._features_schema)

    @property
    def query_id_feature(self) -> FeatureInfo:
        """
        :returns: sequence of query id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.QUERY_ID).item()

    @property
    def item_id_feature(self) -> FeatureInfo:
        """
        :returns: sequence of item id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.ITEM_ID).item()

    @property
    def query_id_column(self) -> str:
        """
        :returns: query id column name.
        """
        return self.query_id_feature.column

    @property
    def item_id_column(self) -> str:
        """
        :returns: item id column name.
        """
        return self.item_id_feature.column

    @property
    def interactions_rating_column(self) -> Optional[str]:
        """
        :returns: interactions-rating column name.
        """
        interactions_rating_features = self.interactions_rating_features
        if not interactions_rating_features:
            return None
        return interactions_rating_features.item().column

    @property
    def interactions_timestamp_column(self) -> Optional[str]:
        """
        :returns: interactions-timestamp column name.
        """
        interactions_timestamp_features = self.interactions_timestamp_features
        if not interactions_timestamp_features:
            return None
        return interactions_timestamp_features.item().column

    def filter(
        self,
        column: Optional[str] = None,
        feature_hint: Optional[FeatureHint] = None,
        feature_source: Optional[FeatureSource] = None,
        feature_type: Optional[FeatureType] = None,
    ) -> "FeatureSchema":
        """Filter list by ``column``, ``feature_source``, ``feature_type`` and ``feature_hint``.

        :param column: Column name to filter by.
            default: ``None``.
        :param feature_hint: Feature hint to filter by.
            default: ``None``.
        :param feature_source: Feature source to filter by.
            default: ``None``.
        :param feature_type: Feature type to filter by.
            default: ``None``.

        :returns: new filtered feature schema.
        """
        filtered_features = self.all_features
        filter_functions = [self._name_filter, self._source_filter, self._type_filter, self._hint_filter]
        filter_parameters = [column, feature_source, feature_type, feature_hint]
        for filtration_func, filtration_param in zip(filter_functions, filter_parameters):
            filtered_features = list(
                filter(
                    lambda x: filtration_func(x, filtration_param),  # type: ignore  # pylint: disable=W0640
                    filtered_features,
                )
            )

        return FeatureSchema(filtered_features)

    def drop(
        self,
        column: Optional[str] = None,
        feature_hint: Optional[FeatureHint] = None,
        feature_source: Optional[FeatureSource] = None,
        feature_type: Optional[FeatureType] = None,
    ) -> "FeatureSchema":
        """Drop features from list by ``column``, ``feature_source``, ``feature_type`` and ``feature_hint``.

        :param column: Column name to filter by.
            default: ``None``.
        :param feature_hint: Feature hint to filter by.
            default: ``None``.
        :param feature_source: Feature source to filter by.
            default: ``None``.
        :param feature_type: Feature type to filter by.
            default: ``None``.

        :returns: new filtered feature schema without selected features.
        """
        filtered_features = self.all_features
        filter_functions = [self._name_drop, self._source_drop, self._type_drop, self._hint_drop]
        filter_parameters = [column, feature_source, feature_type, feature_hint]
        for filtration_func, filtration_param in zip(filter_functions, filter_parameters):
            filtered_features = list(
                filter(
                    lambda x: filtration_func(x, filtration_param),  # type: ignore  # pylint: disable=W0640
                    filtered_features,
                )
            )

        return FeatureSchema(filtered_features)

    @staticmethod
    def _name_filter(value: FeatureInfo, column: str) -> bool:
        return value.column == column if column else True

    @staticmethod
    def _source_filter(value: FeatureInfo, feature_source: FeatureSource) -> bool:
        return value.feature_source == feature_source if feature_source else True

    @staticmethod
    def _type_filter(value: FeatureInfo, feature_type: FeatureType) -> bool:
        return value.feature_type == feature_type if feature_type else True

    @staticmethod
    def _hint_filter(value: FeatureInfo, feature_hint: FeatureHint) -> bool:
        return value.feature_hint == feature_hint if feature_hint else True

    @staticmethod
    def _name_drop(value: FeatureInfo, column: str) -> bool:
        return value.column != column if column else True

    @staticmethod
    def _source_drop(value: FeatureInfo, feature_source: FeatureSource) -> bool:
        return value.feature_source != feature_source if feature_source else True

    @staticmethod
    def _type_drop(value: FeatureInfo, feature_type: FeatureType) -> bool:
        return value.feature_type != feature_type if feature_type else True

    # pylint: disable=no-self-use
    @staticmethod
    def _hint_drop(value: FeatureInfo, feature_hint: FeatureHint) -> bool:
        return value.feature_hint != feature_hint if feature_hint else True

    def _check_features_naming(self, features_list: Sequence[FeatureInfo]) -> None:
        """
        Checks that all the columns have unique names except QUERY_ID and ITEM_ID columns.
        """
        unique_columns = set()
        duplicates = set()
        item_query_names: Dict[FeatureHint, List[str]] = {
            FeatureHint.ITEM_ID: [],
            FeatureHint.QUERY_ID: [],
        }
        for feature in features_list:
            if feature.feature_hint not in [FeatureHint.ITEM_ID, FeatureHint.QUERY_ID]:
                if feature.column in unique_columns:
                    duplicates.add(feature.column)
                else:
                    unique_columns.add(feature.column)
            else:
                item_query_names[feature.feature_hint] += [feature.column]

        if len(duplicates) > 0:
            raise ValueError(
                "Features column names should be unique, exept ITEM_ID and QUERY_ID columns. "
                + f"{duplicates} columns are not unique."
            )

        if len(item_query_names[FeatureHint.ITEM_ID]) > 1:
            raise ValueError(f"ITEM_ID must be present only once. Rename {item_query_names[FeatureHint.ITEM_ID]}")

        if len(item_query_names[FeatureHint.QUERY_ID]) > 1:
            raise ValueError(f"QUERY_ID must be present only once. Rename {item_query_names[FeatureHint.QUERY_ID]}")
