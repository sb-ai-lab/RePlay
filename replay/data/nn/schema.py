from typing import (
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

import torch

from replay.data import FeatureHint, FeatureSource, FeatureType

# Alias
TensorMap = Mapping[str, torch.Tensor]
MutableTensorMap = Dict[str, torch.Tensor]


# pylint: disable=too-many-instance-attributes
class TensorFeatureSource:
    """
    Describes source of a feature
    """

    def __init__(
        self,
        source: FeatureSource,
        column: str,
        index: Optional[int] = None,
    ) -> None:
        """
        :param source: feature source
        :param column: column name
        :param index: index of column in dataframe to get tensor
            directly, without mappings
        """
        self._column = column
        self._index = index
        self._source = source

    @property
    def source(self) -> FeatureSource:
        """
        :returns: feature source
        """
        return self._source

    @property
    def column(self) -> str:
        """
        :returns: column name
        """
        return self._column

    @property
    def index(self) -> Optional[int]:
        """
        :returns: provided index
        """
        return self._index


class TensorFeatureInfo:
    """
    Information about a tensor feature.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name: str,
        feature_type: FeatureType,
        is_seq: bool = False,
        feature_hint: Optional[FeatureHint] = None,
        feature_sources: Optional[List[TensorFeatureSource]] = None,
        cardinality: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        tensor_dim: Optional[int] = None,
    ) -> None:
        """
        :param name: name of feature.
        :param feature_type: type of feature.
        :param is_seq: flag that feature is sequential.
            default: ``False``.
        :param feature_hint: hint to models about feature
            (is timestamp, is rating, is query_id, is item_id),
            default: ``None``.
        :param feature_sources: columns names and DataFrames feature came from,
            default: ``None``.
        :param cardinality: cardinality of categorical feature, required for ids columns,
            optional for others,
            default: ``None``.
        :param embedding_dim: embedding dimensions of categorical feature,
            default: ``None``.
        :param tensor_dim: tensor dimensions of numerical feature,
            default: ``None``.
        """
        self._name = name
        self._feature_hint = feature_hint
        self._feature_sources = feature_sources
        self._is_seq = is_seq

        if not isinstance(feature_type, FeatureType):
            raise ValueError("Unknown feature type")
        self._feature_type = feature_type

        if feature_type == FeatureType.NUMERICAL and (cardinality or embedding_dim):
            raise ValueError("Cardinality and embedding dimensions are needed only with categorical feature type.")
        self._cardinality = cardinality

        if feature_type == FeatureType.CATEGORICAL and tensor_dim:
            raise ValueError("Tensor dimensions is needed only with numerical feature type.")

        if feature_type == FeatureType.CATEGORICAL:
            default_embedding_dim = 64
            self._embedding_dim = embedding_dim or default_embedding_dim
        else:
            self._tensor_dim = tensor_dim

    @property
    def name(self) -> str:
        """
        :returns: The feature name.
        """
        return self._name

    @property
    def feature_type(self) -> FeatureType:
        """
        :returns: The type of feature.
        """
        return self._feature_type

    @property
    def feature_hint(self) -> Optional[FeatureHint]:
        """
        :returns: The feature hint.
        """
        return self._feature_hint

    def _set_feature_hint(self, hint: FeatureHint) -> None:
        self._feature_hint = hint

    @property
    def feature_sources(self) -> Optional[List[TensorFeatureSource]]:
        """
        :returns: List of sources feature came from.
        """
        return self._feature_sources

    def _set_feature_sources(self, sources: List[TensorFeatureSource]) -> None:
        self._feature_sources = sources

    @property
    def feature_source(self) -> Optional[TensorFeatureSource]:
        """
        :returns: Dataframe info of feature.
        """
        source = self.feature_sources
        if not source:
            return None

        if len(source) > 1:
            raise ValueError("Only one element feature sources can be converted to single feature source.")
        assert isinstance(self.feature_sources, list)
        return self.feature_sources[0]

    @property
    def is_seq(self) -> bool:
        """
        :returns: Flag that feature is sequential.
        """
        return self._is_seq

    @property
    def is_cat(self) -> bool:
        """
        :returns: Flag that feature is categorical.
        """
        return self.feature_type == FeatureType.CATEGORICAL

    @property
    def is_num(self) -> bool:
        """
        :returns: Flag that feature is numerical.
        """
        return self.feature_type == FeatureType.NUMERICAL

    @property
    def cardinality(self) -> Optional[int]:
        """
        :returns: Cardinality of the feature.
        """
        if self.feature_type != FeatureType.CATEGORICAL:
            raise RuntimeError(
                f"Can not get cardinality because feature type of {self.name} column is not categorical."
            )
        return self._cardinality

    def _set_cardinality(self, cardinality: int) -> None:
        self._cardinality = cardinality

    @property
    def tensor_dim(self) -> Optional[int]:
        """
        :returns: Dimensions of the numerical feature.
        """
        if self.feature_type != FeatureType.NUMERICAL:
            raise RuntimeError(
                f"Can not get tensor dimensions because feature type of {self.name} feature is not numerical."
            )
        return self._tensor_dim

    def _set_tensor_dim(self, tensor_dim: int) -> None:
        self._tensor_dim = tensor_dim

    @property
    def embedding_dim(self) -> Optional[int]:
        """
        :returns: Embedding dimensions of the feature.
        """
        if self.feature_type != FeatureType.CATEGORICAL:
            raise RuntimeError(
                f"Can not get embedding dimensions because feature type of {self.name} feature is not categorical."
            )
        return self._embedding_dim

    def _set_embedding_dim(self, embedding_dim: int) -> None:
        self._embedding_dim = embedding_dim


class TensorSchema(Mapping[str, TensorFeatureInfo]):
    """
    Key-value like collection that stores tensor features
    """

    def __init__(self, features_list: Union[Sequence[TensorFeatureInfo], TensorFeatureInfo]) -> None:
        """
        :param features_list: list of tensor feature infos.
        """
        features_list = [features_list] if not isinstance(features_list, Sequence) else features_list
        self._tensor_schema = {feature.name: feature for feature in features_list}

    def subset(self, features_to_keep: Iterable[str]) -> "TensorSchema":
        """Creates a subset of given features.

        :param features_to_keep: A sequence of feature names
                in original schema to keep in subset.

        :returns: New tensor schema of given features.
        """
        features: Set[TensorFeatureInfo] = set()
        for feature_name in features_to_keep:
            features.add(self._tensor_schema[feature_name])
        return TensorSchema(list(features))

    def item(self) -> TensorFeatureInfo:
        """
        :returns: Extract single feature from a schema.
        """
        if len(self._tensor_schema) != 1:
            raise ValueError("Only one element tensor schema can be converted to single feature")
        return list(self._tensor_schema.values())[0]

    def items(self) -> ItemsView[str, TensorFeatureInfo]:
        return self._tensor_schema.items()

    def keys(self) -> KeysView[str]:
        return self._tensor_schema.keys()

    def values(self) -> ValuesView[TensorFeatureInfo]:
        return self._tensor_schema.values()

    def get(  # type: ignore
        self,
        key: str,
        default: Optional[TensorFeatureInfo] = None,
    ) -> Optional[TensorFeatureInfo]:
        return self._tensor_schema.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensor_schema)

    def __contains__(self, feature_name: object) -> bool:
        return feature_name in self._tensor_schema

    def __len__(self) -> int:
        return len(self._tensor_schema)

    def __getitem__(self, feature_name: str) -> TensorFeatureInfo:
        return self._tensor_schema[feature_name]

    def __eq__(self, other: object) -> bool:
        return self._tensor_schema == other

    def __ne__(self, other: object) -> bool:
        return self._tensor_schema != other

    def __add__(self, other: "TensorSchema") -> "TensorSchema":
        return TensorSchema(list(self._tensor_schema.values()) + list(other._tensor_schema.values()))

    @property
    def all_features(self) -> Sequence[TensorFeatureInfo]:
        """
        :returns: Sequence of all features.
        """
        return list(self._tensor_schema.values())

    @property
    def categorical_features(self) -> "TensorSchema":
        """
        :returns: Sequence of categorical features in a schema.
        """
        return self.filter(feature_type=FeatureType.CATEGORICAL)

    @property
    def numerical_features(self) -> "TensorSchema":
        """
        :returns: Sequence of numerical features in a schema.
        """
        return self.filter(feature_type=FeatureType.NUMERICAL)

    @property
    def query_id_features(self) -> "TensorSchema":
        """
        :returns: Sequence of query id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.QUERY_ID)

    @property
    def item_id_features(self) -> "TensorSchema":
        """
        :returns: Sequence of item id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.ITEM_ID)

    @property
    def timestamp_features(self) -> "TensorSchema":
        """
        :returns: Sequence of timestamp features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.TIMESTAMP)

    @property
    def rating_features(self) -> "TensorSchema":
        """
        :returns: Sequence of rating features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.RATING)

    @property
    def sequential_features(self) -> "TensorSchema":
        """
        :returns: Sequence of sequential features in a schema.
        """
        return self.filter(is_seq=True)

    @property
    def names(self) -> Sequence[str]:
        """
       :returns: List of all feature's names.
        """
        return list(self._tensor_schema)

    @property
    def query_id_feature_name(self) -> Optional[str]:
        """
        :returns: Query id feature name.
        """
        query_id_features = self.query_id_features
        if not query_id_features:
            return None
        return query_id_features.item().name

    @property
    def item_id_feature_name(self) -> Optional[str]:
        """
        :returns: Item id feature name.
        """
        item_id_features = self.item_id_features
        if not item_id_features:
            return None
        return item_id_features.item().name

    @property
    def timestamp_feature_name(self) -> Optional[str]:
        """
        :returns: Timestamp feature name.
        """
        timestamp_features = self.timestamp_features
        if not timestamp_features:
            return None
        return timestamp_features.item().name

    @property
    def rating_feature_name(self) -> Optional[str]:
        """
        :returns: Rating feature name.
        """
        rating_features = self.rating_features
        if not rating_features:
            return None
        return rating_features.item().name

    def filter(
        self,
        name: Optional[str] = None,
        feature_hint: Optional[FeatureHint] = None,
        is_seq: Optional[bool] = None,
        feature_type: Optional[FeatureType] = None,
    ) -> "TensorSchema":
        """Filter list by ``name``, ``feature_type``, ``is_seq`` and ``feature_hint``.

        :param name: Feature name to filter by.
            default: ``None``.
        :param feature_hint: Feature hint to filter by.
            default: ``None``.
        :param feature_source: Feature source to filter by.
            default: ``None``.
        :param feature_type: Feature type to filter by.
            default: ``None``.

        :returns: New filtered feature schema.
        """
        filtered_features = self.all_features
        filter_functions = [self._name_filter, self._seq_filter, self._type_filter, self._hint_filter]
        filter_parameters = [name, is_seq, feature_type, feature_hint]
        for filtration_func, filtration_param in zip(filter_functions, filter_parameters):
            filtered_features = list(
                filter(
                    lambda x: filtration_func(x, filtration_param),  # type: ignore  # pylint: disable=W0640
                    filtered_features,
                )
            )

        return TensorSchema(filtered_features)

    @staticmethod
    def _name_filter(value: TensorFeatureInfo, name: str) -> bool:
        return value.name == name if name else True

    @staticmethod
    def _seq_filter(value: TensorFeatureInfo, is_seq: bool) -> bool:
        return value.is_seq == is_seq if is_seq is not None else True

    @staticmethod
    def _type_filter(value: TensorFeatureInfo, feature_type: FeatureType) -> bool:
        return value.feature_type == feature_type if feature_type else True

    @staticmethod
    def _hint_filter(value: TensorFeatureInfo, feature_hint: FeatureHint) -> bool:
        return value.feature_hint == feature_hint if feature_hint else True
