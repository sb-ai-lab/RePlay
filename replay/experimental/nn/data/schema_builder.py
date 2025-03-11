from typing import Dict, List, Optional

from replay.data import FeatureHint, FeatureType
from replay.data.nn.schema import TensorFeatureInfo, TensorFeatureSource, TensorSchema


class TensorSchemaBuilder:
    """
    Builder that simplifies creating tensor schema
    """

    def __init__(self) -> None:
        self._tensor_schema: Dict[str, TensorFeatureInfo] = {}

    def categorical(
        self,
        name: str,
        cardinality: int,
        is_seq: bool = False,
        feature_source: Optional[TensorFeatureSource] = None,
        feature_hint: Optional[FeatureHint] = None,
        embedding_dim: Optional[int] = None,
        padding_value: int = 0,
    ) -> "TensorSchemaBuilder":
        source = [feature_source] if feature_source else None
        self._tensor_schema[name] = TensorFeatureInfo(
            name=name,
            feature_type=FeatureType.CATEGORICAL,
            is_seq=is_seq,
            feature_sources=source,
            feature_hint=feature_hint,
            cardinality=cardinality,
            padding_value=padding_value,
            embedding_dim=embedding_dim,
        )
        return self

    def numerical(
        self,
        name: str,
        tensor_dim: int,
        is_seq: bool = False,
        feature_sources: Optional[List[TensorFeatureSource]] = None,
        feature_hint: Optional[FeatureHint] = None,
        padding_value: int = 0,
    ) -> "TensorSchemaBuilder":
        self._tensor_schema[name] = TensorFeatureInfo(
            name=name,
            feature_type=FeatureType.NUMERICAL,
            is_seq=is_seq,
            feature_sources=feature_sources,
            feature_hint=feature_hint,
            tensor_dim=tensor_dim,
            padding_value=padding_value,
        )
        return self

    def categorical_list(
        self,
        name: str,
        cardinality: int,
        is_seq: bool = False,
        feature_source: Optional[TensorFeatureSource] = None,
        feature_hint: Optional[FeatureHint] = None,
        embedding_dim: Optional[int] = None,
        padding_value: int = 0,
    ) -> "TensorSchemaBuilder":
        source = [feature_source] if feature_source else None
        self._tensor_schema[name] = TensorFeatureInfo(
            name=name,
            feature_type=FeatureType.CATEGORICAL_LIST,
            is_seq=is_seq,
            feature_sources=source,
            feature_hint=feature_hint,
            cardinality=cardinality,
            padding_value=padding_value,
            embedding_dim=embedding_dim,
        )
        return self

    def numerical_list(
        self,
        name: str,
        tensor_dim: int,
        is_seq: bool = False,
        feature_sources: Optional[List[TensorFeatureSource]] = None,
        feature_hint: Optional[FeatureHint] = None,
        padding_value: int = 0,
    ) -> "TensorSchemaBuilder":
        self._tensor_schema[name] = TensorFeatureInfo(
            name=name,
            feature_type=FeatureType.NUMERICAL_LIST,
            is_seq=is_seq,
            feature_sources=feature_sources,
            feature_hint=feature_hint,
            tensor_dim=tensor_dim,
            padding_value=padding_value,
        )
        return self

    def build(self) -> TensorSchema:
        return TensorSchema(list(self._tensor_schema.values()))
