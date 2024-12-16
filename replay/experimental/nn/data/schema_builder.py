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
    ) -> "TensorSchemaBuilder":
        source = [feature_source] if feature_source else None
        self._tensor_schema[name] = TensorFeatureInfo(
            name=name,
            feature_type=FeatureType.CATEGORICAL,
            is_seq=is_seq,
            feature_sources=source,
            feature_hint=feature_hint,
            cardinality=cardinality,
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
    ) -> "TensorSchemaBuilder":
        self._tensor_schema[name] = TensorFeatureInfo(
            name=name,
            feature_type=FeatureType.NUMERICAL,
            is_seq=is_seq,
            feature_sources=feature_sources,
            feature_hint=feature_hint,
            tensor_dim=tensor_dim,
        )
        return self

    def build(self) -> TensorSchema:
        return TensorSchema(list(self._tensor_schema.values()))
