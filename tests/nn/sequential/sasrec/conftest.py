import pytest
import torch

from replay.data.nn import ParquetModule
from replay.data.nn.schema import TensorSchema
from replay.nn.agg import ConcatAggregator
from replay.nn.embedding import SequenceEmbedding
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.mask import DefaultAttentionMask
from replay.nn.sequential import DiffTransformerLayer, PositionAwareAggregator, SasRec, SasRecBody
from replay.nn.transform.template import make_default_sasrec_transforms


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
def sasrec_parametrized(request, tensor_schema):
    loss_cls, kwargs = request.param
    loss = loss_cls(**kwargs)
    model_hidden_size = tensor_schema["item_id"].embedding_dim
    body = SasRecBody(
        embedder=SequenceEmbedding(tensor_schema, categorical_list_feature_aggregation_method="sum"),
        embedding_aggregator=(
            PositionAwareAggregator(
                ConcatAggregator(
                    input_embedding_dims=[x.embedding_dim for x in tensor_schema.values()],
                    output_embedding_dim=model_hidden_size,
                ),
                max_sequence_length=7,
                dropout=0.2,
            )
        ),
        attn_mask_builder=DefaultAttentionMask("item_id", 1),
        encoder=DiffTransformerLayer(embedding_dim=model_hidden_size, num_heads=1, num_blocks=1),
        output_normalization=torch.nn.LayerNorm(model_hidden_size),
    )
    model = SasRec(body=body, loss=loss)
    return model


@pytest.fixture
def sasrec_model_only_items(tensor_schema_with_equal_embedding_dims):
    model = SasRec.from_params(
        schema=tensor_schema_with_equal_embedding_dims.filter(name="item_id"),
        embedding_dim=14,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


def create_shared_meta(shape: int, tensor_schema: TensorSchema):
    shared_meta = {
        "user_id": {},
        "item_id": {"shape": shape, "padding": tensor_schema["item_id"].padding_value},
    }
    return shared_meta


@pytest.fixture
def parquet_module_with_default_sasrec_transform(
    parquet_module_path, tensor_schema_with_equal_embedding_dims, parquet_module_config, max_len, batch_size=4
):
    transforms = make_default_sasrec_transforms(tensor_schema_with_equal_embedding_dims)

    tensor_schema = tensor_schema_with_equal_embedding_dims
    metadata = {
        "train": create_shared_meta(shape=max_len + 1, tensor_schema=tensor_schema),
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
def parquet_module_with_multiple_val_paths(parquet_module_path, tensor_schema, max_len, batch_size=4):
    transforms = make_default_sasrec_transforms(tensor_schema)

    metadata = {
        "train": create_shared_meta(shape=max_len + 1, tensor_schema=tensor_schema),
        "validate": create_shared_meta(shape=max_len, tensor_schema=tensor_schema),
        "test": create_shared_meta(shape=max_len, tensor_schema=tensor_schema),
        "predict": create_shared_meta(shape=max_len, tensor_schema=tensor_schema),
    }

    parquet_module = ParquetModule(
        metadata=metadata,
        transforms=transforms,
        batch_size=batch_size,
        train_path=parquet_module_path,
        validate_path=[parquet_module_path, parquet_module_path],
        test_path=parquet_module_path,
        predict_path=parquet_module_path,
    )
    return parquet_module
