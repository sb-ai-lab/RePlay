import copy

import pytest
import torch

from replay.nn.sequential.bert4rec import Bert4Rec


@pytest.fixture
def bert4rec_model(tensor_schema_with_equal_embedding_dims):
    schema = tensor_schema_with_equal_embedding_dims.filter(name="item_id")
    model = Bert4Rec.from_params(
        schema=schema,
        embedding_dim=schema["item_id"].embedding_dim,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


@pytest.fixture
def bert4rec_sequential_sample(tensor_schema_with_equal_embedding_dims):
    schema = tensor_schema_with_equal_embedding_dims.filter(name="item_id")
    padding_value = schema["item_id"].padding_value
    item_sequences = torch.LongTensor(
        [
            [padding_value, padding_value, padding_value, 1, 2, 3, 4],
            [padding_value, padding_value, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5, 6],
            [padding_value, padding_value, padding_value, padding_value, 2, 1, 0],
        ],
    )
    padding_mask = item_sequences != padding_value
    token_mask = torch.BoolTensor(
        [
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
        ],
    )
    target_padding_mask = torch.logical_xor(padding_mask, token_mask)

    return {
        "feature_tensors": {
            "item_id": item_sequences,
            "token_mask": token_mask,
        },
        "padding_mask": padding_mask,
        "positive_labels": item_sequences.unsqueeze(-1),
        "target_padding_mask": target_padding_mask.unsqueeze(-1),
        "query_id": torch.LongTensor([0, 1, 2, 3]),
    }


@pytest.fixture
def wrong_bert4rec_sequential_sample(request, bert4rec_sequential_sample):
    sample = copy.deepcopy(bert4rec_sequential_sample)

    defect_type = request.param
    if defect_type == "missing field":
        sample["feature_tensors"].pop("token_mask")
    elif defect_type == "wrong length":
        sample["feature_tensors"]["item_id"] = sample["feature_tensors"]["item_id"][:, 1:]
    elif defect_type == "index out of embedding":
        sample["feature_tensors"]["item_id"][0][-1] = 16
    else:
        raise ValueError(defect_type)
    return sample
