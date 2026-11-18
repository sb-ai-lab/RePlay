"""Checkpoint config capture: rebuild a module from a saved artifact (D8)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema

CONFIG_SUFFIX = ".config.json"


@dataclass
class ModelConfig:
    """Everything needed to rebuild a model next to its checkpoint."""

    model_type: str
    from_params: dict[str, Any]
    schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            model_type=data["model_type"],
            from_params=data["from_params"],
            schema=data["schema"],
        )


def _feature_info_to_dict(info: TensorFeatureInfo) -> dict[str, Any]:
    return {
        "name": info.name,
        "feature_type": info.feature_type.value,
        "is_seq": info.is_seq,
        "feature_hint": info.feature_hint.value if info.feature_hint is not None else None,
        "feature_sources": [
            {
                "source": source.source.value,
                "column": source.column,
                "index": source.index,
            }
            for source in info.feature_sources or []
        ],
        "cardinality": info.cardinality if info.is_cat else None,
        "padding_value": info.padding_value,
        "embedding_dim": info.embedding_dim,
        "tensor_dim": info.tensor_dim if info.is_num else None,
    }


def _feature_info_from_dict(data: dict[str, Any]) -> TensorFeatureInfo:
    return TensorFeatureInfo(
        name=data["name"],
        feature_type=FeatureType(data["feature_type"]),
        is_seq=data["is_seq"],
        feature_hint=FeatureHint(data["feature_hint"]) if data["feature_hint"] is not None else None,
        feature_sources=[
            TensorFeatureSource(
                source=FeatureSource(source["source"]),
                column=source["column"],
                index=source["index"],
            )
            for source in data["feature_sources"]
        ],
        cardinality=data["cardinality"],
        padding_value=data["padding_value"],
        embedding_dim=data["embedding_dim"],
        tensor_dim=data["tensor_dim"],
    )


def schema_to_dict(schema: TensorSchema) -> dict[str, Any]:
    """Serialize a ``TensorSchema`` into JSON-primitive form."""
    return {"features": [_feature_info_to_dict(info) for info in schema.values()]}


def schema_from_dict(data: dict[str, Any]) -> TensorSchema:
    """Rebuild a ``TensorSchema`` from :func:`schema_to_dict` output."""
    return TensorSchema([_feature_info_from_dict(feature) for feature in data["features"]])


def capture_config(model_cls, schema: TensorSchema, **from_params: Any) -> tuple[torch.nn.Module, ModelConfig]:
    """Build a model with ``model_cls.from_params`` and capture its rebuild recipe.

    :param model_cls: a model class such as ``SasRec`` exposing ``from_params``.
    :param schema: the model's ``TensorSchema``.
    :param from_params: keyword arguments forwarded to ``from_params``;
        they must be JSON-primitive (the schema is stored separately).
    :return: the built model and its :class:`ModelConfig`.
    """
    from ._adapters import resolve_adapter

    try:
        json.dumps(from_params)
    except TypeError as exc:
        msg = f"from_params kwargs must be JSON-primitive to be captured: {exc}"
        raise ValueError(msg) from None
    model = model_cls.from_params(schema=schema, **from_params)
    config = ModelConfig(
        model_type=resolve_adapter(model).model_type,
        from_params=from_params,
        schema=schema_to_dict(schema),
    )
    return model, config


def config_path_for(ckpt_path: str | Path) -> Path:
    """Sidecar path paired with ``ckpt_path`` (``<ckpt>.config.json``)."""
    return Path(f"{ckpt_path}{CONFIG_SUFFIX}")


def save_config(config: ModelConfig, ckpt_path: str | Path) -> Path:
    """Write the config sidecar next to the checkpoint artifact."""
    path = config_path_for(ckpt_path)
    path.write_text(json.dumps(config.to_dict(), indent=2))
    return path


def read_config(ckpt_path: str | Path) -> ModelConfig:
    """Read the config sidecar paired with ``ckpt_path``."""
    path = config_path_for(ckpt_path)
    if not path.is_file():
        msg = f"No config sidecar for checkpoint {ckpt_path!s}: expected {path}"
        raise FileNotFoundError(msg)
    return ModelConfig.from_dict(json.loads(path.read_text()))


def _model_class(model_type: str):
    from replay.nn.sequential.sasrec.model import SasRec

    from ._adapters import SasRecAdapter

    if model_type == SasRecAdapter.model_type:
        return SasRec
    msg = f"No rebuild support for model type: {model_type!r}"
    raise ValueError(msg)


def _load_state_dict(ckpt_path: Path) -> dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {key.removeprefix("model."): value for key, value in state_dict.items()}


def restore_from_checkpoint(
    ckpt_path: str | Path, config_path: str | Path | None = None
) -> tuple[torch.nn.Module, TensorSchema]:
    """Rebuild a model and its schema from a checkpoint plus config sidecar.

    :param ckpt_path: checkpoint file: a raw ``state_dict`` or a Lightning
        checkpoint (its ``state_dict`` is used, ``model.`` prefixes stripped).
    :param config_path: explicit sidecar path; ``None`` reads the sibling
        ``<ckpt>.config.json``.
    :return: the rebuilt module (weights loaded, ``eval`` mode) and its schema.
    """
    path = Path(ckpt_path)
    if config_path is not None:
        config = ModelConfig.from_dict(json.loads(Path(config_path).read_text()))
    else:
        config = read_config(path)
    schema = schema_from_dict(config.schema)
    model = _model_class(config.model_type).from_params(schema=schema, **config.from_params)
    model.load_state_dict(_load_state_dict(path), strict=True)
    model.eval()
    return model, schema
