# Preserve tensor feature padding values during serialization

## Problem

`TensorSchema._get_object_args()` serialized tensor feature metadata without including
`TensorFeatureInfo.padding_value`. When a schema was restored, the constructor silently
used its default padding value of `0`. Any custom value, such as `-1`, was therefore lost
when saving and loading sequential datasets or sequence tokenizers.

## Fix

The serialized representation now includes `padding_value` for every tensor feature.
`TensorSchema._create_object_by_args()` already passes serialized fields to
`TensorFeatureInfo`, so no separate deserialization logic is required. Older saved data
without this field remains compatible because `TensorFeatureInfo` defaults it to `0`.

## Regression coverage

The tests verify two levels of behavior:

- A `TensorSchema` metadata round trip preserves a custom padding value.
- Saving and loading sequential datasets preserves the padding value of every feature.

Run the focused checks with:

```bash
pytest tests/data/nn/test_schema.py tests/data/nn/test_sequential_dataset.py
```
