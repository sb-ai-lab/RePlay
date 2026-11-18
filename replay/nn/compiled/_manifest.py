"""Export manifest: everything a consumer needs to run a compiled graph."""

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

FEATURE_INPUT = "feature"
PADDING_MASK_INPUT = "padding_mask"
CANDIDATES_INPUT = "candidates"

CANDIDATES_FULL_VOCAB = "full_vocab"
CANDIDATES_EXPLICIT = "explicit"

MANIFEST_FILENAME = "manifest.json"

_TORCH_TO_NUMPY_DTYPE = {
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float32: "float32",
    torch.float64: "float64",
}


def numpy_dtype_name(dtype: torch.dtype) -> str:
    try:
        return _TORCH_TO_NUMPY_DTYPE[dtype]
    except KeyError:
        msg = f"Unsupported dtype for compiled graphs: {dtype}"
        raise ValueError(msg) from None


@dataclass
class FeatureRecord:
    """Schema facts needed to validate and encode one graph input."""

    name: str
    dtype: str
    is_categorical: bool
    is_list: bool
    cardinality: int | None = None
    tensor_dim: int | None = None
    embedding_dim: int | None = None
    padding_value: int | None = None

    @classmethod
    def from_tensor_info(cls, info) -> "FeatureRecord":
        dtype = torch.int64 if info.is_cat else torch.float32
        return cls(
            name=info.name,
            dtype=numpy_dtype_name(dtype),
            is_categorical=info.is_cat,
            is_list=info.is_list,
            cardinality=info.cardinality if info.is_cat else None,
            tensor_dim=info.tensor_dim if info.is_num else None,
            embedding_dim=info.embedding_dim,
            padding_value=info.padding_value if info.is_cat else None,
        )


@dataclass
class ExportManifest:
    """Positional-input contract of one exported graph.

    The graph inputs are always ``feature`` tensors in schema order,
    then ``padding_mask``, then ``candidates`` if the policy is explicit.
    """

    model_type: str
    exporter: str
    opset_version: int
    input_kinds: list[str]
    input_names: list[str]
    output_names: list[str]
    candidates_policy: str
    features: list[FeatureRecord]
    dynamic_axes: dict[str, list[str]] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)

    def feature_names(self) -> list[str]:
        return [record.name for record in self.features]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = [asdict(record) for record in self.features]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportManifest":
        features = [FeatureRecord(**record) for record in data.pop("features")]
        return cls(features=features, **data)
