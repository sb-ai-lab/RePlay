import copy

import pytest
import torch

from replay.data.nn import ParquetModule
from replay.data.nn.schema import TensorSchema
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.sequential.bert4rec import BERT4Rec
from replay.nn.transform import UniformNegativeSamplingTransform
from replay.nn.transform.template import make_default_bert4rec_transforms


@pytest.fixture(
    params=[
        (CE, {"ignore_index": 15}),
        (CESampled, {"ignore_index": 15}),
        (CESampled, {"ignore_index": 15, "negative_labels_ignore_index": 0}),
        (BCE, {}),
        (BCESampled, {}),
        (LogOutCE, {"ignore_index": 15, "cardinality": 15}),
        (LogInCE, {"cardinality": 15}),
        (LogInCE, {"cardinality": 15, "negative_labels_ignore_index": 0}),
        (LogInCESampled, {}),
        (LogInCESampled, {"negative_labels_ignore_index": 0}),
    ],
    ids=[
        "CE",
        "CE sampled",
        "CE sampled w/ negative ignore index",
        "BCE",
        "BCE sampled",
        "LogOutCE",
        "LogInCE",
        "LogInCE w/ negative ignore index",
        "LogInCESampled",
        "LogInCESampled w/ negative ignore index",
    ],
)
def bert4rec_parametrized(request, tensor_schema_with_equal_embedding_dims):
    loss_cls, kwargs = request.param
    loss = loss_cls(**kwargs)

    model = BERT4Rec.from_params(
        schema=tensor_schema_with_equal_embedding_dims,
        embedding_dim=tensor_schema_with_equal_embedding_dims["item_id"].embedding_dim,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    model.loss = loss
    model.loss.logits_callback = model.get_logits
    return model


@pytest.fixture
def bert4rec_model(tensor_schema_with_equal_embedding_dims):
    model = BERT4Rec.from_params(
        schema=tensor_schema_with_equal_embedding_dims,
        embedding_dim=tensor_schema_with_equal_embedding_dims["item_id"].embedding_dim,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


@pytest.fixture
def bert4rec_model_only_items(tensor_schema_with_equal_embedding_dims):
    schema = tensor_schema_with_equal_embedding_dims.filter(name="item_id")
    model = BERT4Rec.from_params(
        schema=schema,
        embedding_dim=schema["item_id"].embedding_dim,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


@pytest.fixture
def bert4rec_sequential_sample(simple_batch):
    item_sequences = simple_batch["feature_tensors"]["item_id"]
    padding_value = 15
    padding_mask = item_sequences != padding_value
    token_mask = torch.BoolTensor(
        [
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 1, 0, 1],
        ],
    )
    target_padding_mask = torch.logical_xor(padding_mask, token_mask)
    feature_tensors = dict(simple_batch["feature_tensors"])
    feature_tensors["token_mask"] = token_mask

    return {
        "feature_tensors": feature_tensors,
        "padding_mask": padding_mask,
        "positive_labels": item_sequences.unsqueeze(-1),
        "target_padding_mask": target_padding_mask.unsqueeze(-1),
        "query_id": simple_batch["query_id"],
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


def create_shared_meta(shape: int, tensor_schema: TensorSchema):
    shared_meta = {
        "user_id": {},
        "item_id": {"shape": shape, "padding": tensor_schema["item_id"].padding_value},
        "cat_list_feature": {"shape": [shape, 3], "padding": tensor_schema["cat_list_feature"].padding_value},
        "num_feature": {"shape": shape, "padding": tensor_schema["num_feature"].padding_value},
        "num_list_feature": {
            "shape": [shape, tensor_schema["num_list_feature"].tensor_dim],
            "padding": tensor_schema["num_list_feature"].padding_value,
        },
        "emb_list_feature": {
            "shape": [shape, tensor_schema["emb_list_feature"].tensor_dim],
            "padding": tensor_schema["emb_list_feature"].padding_value,
        },
    }
    return shared_meta


@pytest.fixture
def parquet_module_with_default_bert4rec_transform(
    parquet_module_path,
    tensor_schema_with_equal_embedding_dims,
    parquet_module_config,
    max_len,
    batch_size=4,
):
    transforms = make_default_bert4rec_transforms(
        tensor_schema=tensor_schema_with_equal_embedding_dims,
    )

    tensor_schema = tensor_schema_with_equal_embedding_dims
    metadata = {
        "train": create_shared_meta(shape=max_len, tensor_schema=tensor_schema),
        "validate": create_shared_meta(shape=max_len, tensor_schema=tensor_schema),
        "test": create_shared_meta(shape=max_len, tensor_schema=tensor_schema),
        "predict": create_shared_meta(shape=max_len, tensor_schema=tensor_schema),
    }

    parquet_module = ParquetModule(
        metadata=metadata,
        transforms=transforms,
        batch_size=batch_size,
        train_path=parquet_module_path,
        validate_path=parquet_module_path,
        test_path=parquet_module_path,
        predict_path=parquet_module_path,
        config=parquet_module_config,
    )
    return parquet_module


@pytest.fixture
def parquet_module_with_bert4rec_negative_sampling_transform(
    parquet_module_with_default_bert4rec_transform,
    tensor_schema_with_equal_embedding_dims,
):
    transforms = parquet_module_with_default_bert4rec_transform.transforms
    transforms["train"].insert(
        -1,
        UniformNegativeSamplingTransform(
            cardinality=tensor_schema_with_equal_embedding_dims["item_id"].cardinality,
            num_negative_samples=10,
        ),
    )
    parquet_module_with_default_bert4rec_transform.compiled_transforms = (
        parquet_module_with_default_bert4rec_transform.prepare_transforms(transforms)
    )
    return parquet_module_with_default_bert4rec_transform
