"""Pre-inference encoding of online event dicts into compiled-graph inputs.

The contract implemented here is recorded in every export manifest
(``encoding`` block) and mirrors ``TorchSequentialDataset``:
long histories keep their last ``max_seq_len`` events, short ones are
front-padded with the feature's ``padding_value``, and ``padding_mask``
marks valid positions as ``True``.  Unknown categorical values and absent
numerical values also fall back to ``padding_value``.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._manifest import ExportManifest

_SUPPORTED_ENCODING = {
    "padding_side": "front",
    "keep": "last",
    "default_value": "padding_value",
    "mask_true": "valid",
}


def _manifest_data(manifest) -> dict[str, Any]:
    if hasattr(manifest, "to_dict"):
        return manifest.to_dict()
    if isinstance(manifest, dict):
        return manifest
    msg = f"Expected an ExportManifest or its dict form, got: {type(manifest).__name__}"
    raise TypeError(msg)


def _check_encoding(encoding: dict[str, Any]) -> None:
    for key, expected in _SUPPORTED_ENCODING.items():
        actual = encoding.get(key, expected)
        if actual != expected:
            msg = f"Unsupported encoding contract for {key!r}: {actual!r} (helper implements {expected!r})"
            raise ValueError(msg)


def _normalize_batch(batch) -> list[list[dict[str, Any]]]:
    if isinstance(batch, dict):
        batch = batch.get("events")
    if isinstance(batch, dict) or not isinstance(batch, (list, tuple)):
        msg = "Batch must be a list of event dicts, a list of such lists, or {'events': [...]}"
        raise ValueError(msg)
    if not batch:
        msg = "Batch must contain at least one user history"
        raise ValueError(msg)
    if isinstance(batch[0], dict):
        batch = [batch]
    histories = []
    for history in batch:
        if not isinstance(history, (list, tuple)) or not history:
            msg = "Each user history must be a non-empty list of event dicts"
            raise ValueError(msg)
        if not all(isinstance(event, dict) for event in history):
            msg = "Each event must be a dict of {feature_name: raw value}"
            raise ValueError(msg)
        histories.append(list(history))
    return histories


def _static_axis(data: dict[str, Any], name: str, position: int) -> int | None:
    shapes = data.get("input_shapes", {})
    if name not in shapes:
        msg = f"Manifest has no input shape for feature {name!r}; recompile the artifact"
        raise ValueError(msg)
    dim = shapes[name][position]
    return dim if isinstance(dim, int) else None


def _resolve_target(histories, seq_len: int | None, cap: int) -> int:
    if seq_len is not None and seq_len > cap:
        msg = f"seq_len={seq_len} exceeds the manifest max_seq_len={cap}"
        raise ValueError(msg)
    return seq_len or min(max(len(history) for history in histories), cap)


def _encode_categorical(record, values, mapping):
    if mapping is None:
        msg = f"No id mapping for categorical feature {record['name']!r}; pass mappings[{record['name']!r}]"
        raise ValueError(msg)
    pad = record["padding_value"]
    if record["is_list"]:
        return [[mapping.get(x, pad) for x in (row if row is not None else [])] for row in values]
    return [mapping.get(v, pad) for v in values]


def _encode_numerical(record, values):
    pad = record["padding_value"]
    if record["is_list"]:
        return [[float(x) for x in (row if row is not None else [])] for row in values]
    return [float(v) if v is not None else pad for v in values]


def _normalize_rows(rows, width, pad):
    return [[pad] * (width - len(row[:width])) + list(row[:width]) for row in rows]


def _pad_front(rows, target, pad, dtype, width=None):
    shape = (target,) if width is None else (target, width)
    array = np.full(shape, pad, dtype=dtype)
    array[len(array) - len(rows) :] = rows
    return array


def encode_inputs(
    manifest: "ExportManifest | dict[str, Any]",
    batch,
    mappings: dict[str, Any] | None = None,
    *,
    seq_len: int | None = None,
) -> dict[str, np.ndarray]:
    """Encode online event dicts into NumPy inputs for :meth:`CompiledModel.predict`.

    ``batch`` is one user history (a list of ``{feature_name: raw value}`` event
    dicts), a batch of such histories, or a single ``{"events": [...]}`` payload.
    ``mappings`` maps each categorical feature name to an id mapping with a
    ``get(value, default)`` method (e.g. a ``LabelEncoder`` rule mapping);
    unmapped and missing values become the feature's ``padding_value``.

    Per the manifest encoding contract: histories longer than ``max_seq_len``
    keep their most recent events, shorter ones are front-padded, list-feature
    rows are front-padded (and truncated) to the width fixed at compile time,
    and ``padding_mask`` marks valid positions.  ``seq_len`` overrides the pad
    target (the longest history in the batch by default) and must not exceed
    ``max_seq_len``.

    :param manifest: the export manifest (object or parsed ``manifest.json``).
    :param batch: online event histories, one per user.
    :param mappings: id mappings for categorical features.
    :param seq_len: explicit sequence length to pad to.
    :return: arrays keyed by graph input name, ready for ``predict``.
    """
    mappings = mappings or {}
    data = _manifest_data(manifest)
    _check_encoding(data.get("encoding") or {})
    cap = int(data.get("max_seq_len") or 0)
    if cap <= 0:
        msg = "Manifest predates the encoding contract (no max_seq_len); recompile the artifact"
        raise ValueError(msg)

    histories = _normalize_batch(batch)
    target = _resolve_target(histories, seq_len, cap)

    static_batch = _static_axis(data, data["features"][0]["name"], 0)
    if static_batch is not None and len(histories) != static_batch:
        msg = f"Batch of {len(histories)} does not match the fixed batch size {static_batch} of this graph"
        raise ValueError(msg)

    inputs: dict[str, np.ndarray] = {}
    for record in data["features"]:
        name = record["name"]
        if not record.get("is_seq", True):
            msg = f"Feature {name!r} is not sequential; compiled graphs expect sequential features"
            raise NotImplementedError(msg)
        dtype = np.dtype(record["dtype"])
        pad = record["padding_value"]
        width = _static_axis(data, name, 2) if record["is_list"] or (record["tensor_dim"] or 1) > 1 else None
        static_seq = _static_axis(data, name, 1)
        if static_seq is not None and target != static_seq:
            msg = f"Feature {name!r} has a fixed sequence length {static_seq}; encode with seq_len={static_seq}"
            raise ValueError(msg)

        rows_per_user = []
        for history in histories:
            values = [event.get(name) for event in history[-target:]]
            if record["is_categorical"]:
                encoded = _encode_categorical(record, values, mappings.get(name))
            else:
                encoded = _encode_numerical(record, values)
            if width is not None:
                encoded = _normalize_rows(encoded, width, pad)
            rows_per_user.append(encoded)

        inputs[name] = np.stack([_pad_front(rows, target, pad, dtype, width) for rows in rows_per_user])

    mask = np.zeros((len(histories), target), dtype=bool)
    for row, history in enumerate(histories):
        mask[row, target - min(len(history), target) :] = True
    inputs["padding_mask"] = mask
    return inputs
