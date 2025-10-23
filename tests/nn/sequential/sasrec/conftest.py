import pytest
import torch

<<<<<<< HEAD
from replay.data.nn import ParquetModule
from replay.nn.agg import ConcatAggregator
from replay.nn.embedding import SequenceEmbedding
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.mask import DefaultAttentionMask
from replay.nn.sequential import DiffTransformerLayer, PositionAwareAggregator, SasRec, SasRecBody
from replay.nn.transform.template import make_default_sasrec_transforms
=======
from replay.data.nn import ParquetModule, TensorSchema
from replay.nn import ConcatAggregator, DefaultAttentionMask, SequenceEmbedding
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.sequential.sasrec import DiffTransformerLayer, PositionAwareAggregator, SasRec, SasRecBody
from replay.nn.transforms.templates.sasrec import make_default_sasrec_transforms
>>>>>>> Add saving/loading `linucb_arms` in pickle format


@pytest.fixture(
    params=[
<<<<<<< HEAD
        (CE, {"ignore_index": 15}),
        (CESampled, {"ignore_index": 15}),
        (CESampled, {"ignore_index": 15, "negative_labels_ignore_index": 0}),
        (BCE, {}),
        (BCESampled, {}),
        (LogOutCE, {"ignore_index": 15, "cardinality": 15}),
        (LogInCE, {"cardinality": 15}),
        (LogInCE, {"cardinality": 15, "negative_labels_ignore_index": 0}),
=======
        (CE, {"ignore_index": 40}),
        (CESampled, {"ignore_index": 40}),
        (CESampled, {"ignore_index": 40, "negative_labels_ignore_index": 0}),
        (BCE, {}),
        (BCESampled, {}),
        (LogOutCE, {"ignore_index": 40, "cardinality": 40}),
        (LogInCE, {"cardinality": 40}),
        (LogInCE, {"cardinality": 40, "negative_labels_ignore_index": 0}),
>>>>>>> Add saving/loading `linucb_arms` in pickle format
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
<<<<<<< HEAD
    model_hidden_size = tensor_schema["item_id"].embedding_dim
=======

>>>>>>> Add saving/loading `linucb_arms` in pickle format
    body = SasRecBody(
        embedder=SequenceEmbedding(tensor_schema, categorical_list_feature_aggregation_method="sum"),
        embedding_aggregator=(
            PositionAwareAggregator(
                ConcatAggregator(
                    input_embedding_dims=[x.embedding_dim for x in tensor_schema.values()],
<<<<<<< HEAD
                    output_embedding_dim=model_hidden_size,
=======
                    output_embedding_dim=64,
>>>>>>> Add saving/loading `linucb_arms` in pickle format
                ),
                max_sequence_length=7,
                dropout=0.2,
            )
        ),
        attn_mask_builder=DefaultAttentionMask("item_id", 1),
<<<<<<< HEAD
        encoder=DiffTransformerLayer(embedding_dim=model_hidden_size, num_heads=1, num_blocks=1),
        output_normalization=torch.nn.LayerNorm(model_hidden_size),
=======
        encoder=DiffTransformerLayer(embedding_dim=64, num_heads=1, num_blocks=1),
        output_normalization=torch.nn.LayerNorm(64),
>>>>>>> Add saving/loading `linucb_arms` in pickle format
    )
    model = SasRec(body=body, loss=loss)
    return model


@pytest.fixture
def sasrec_model_only_items(tensor_schema_with_equal_embedding_dims):
    model = SasRec.from_params(
        schema=tensor_schema_with_equal_embedding_dims.filter(name="item_id"),
<<<<<<< HEAD
        embedding_dim=14,
=======
        embedding_dim=70,
>>>>>>> Add saving/loading `linucb_arms` in pickle format
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


<<<<<<< HEAD
=======
def create_shared_meta(shape: int, tensor_schema: TensorSchema):
    shared_meta = {
        "user_id": {},
        "item_id": {"shape": shape, "padding": tensor_schema["item_id"].padding_value},
    }
    return shared_meta


>>>>>>> Add saving/loading `linucb_arms` in pickle format
@pytest.fixture
def parquet_module_with_default_sasrec_transform(
    parquet_module_path, tensor_schema_with_equal_embedding_dims, max_len, batch_size=4
):
    transforms = make_default_sasrec_transforms(tensor_schema_with_equal_embedding_dims, query_column="user_id")

<<<<<<< HEAD
    def create_meta(shape):
        shared_meta = {
            "user_id": {},
            "item_id": {"shape": shape, "padding": tensor_schema_with_equal_embedding_dims["item_id"].padding_value},
        }
        return shared_meta

    metadata = {
        "train": create_meta(shape=max_len + 1),
        "validate": create_meta(shape=max_len),
        "test": create_meta(shape=max_len),
        "predict": create_meta(shape=max_len),
=======
    metadata = {
        "train": create_shared_meta(max_len + 1, tensor_schema),
        "validate": create_shared_meta(max_len, tensor_schema),
        "test": create_shared_meta(max_len, tensor_schema),
        "predict": create_shared_meta(max_len, tensor_schema),
>>>>>>> Add saving/loading `linucb_arms` in pickle format
    }

    parquet_module = ParquetModule(
        metadata=metadata,
        transforms=transforms,
        batch_size=batch_size,
        train_path=parquet_module_path,
        validate_path=parquet_module_path,
        test_path=parquet_module_path,
        predict_path=parquet_module_path,
    )
    return parquet_module
<<<<<<< HEAD
=======


@pytest.fixture
def parquet_module_with_multiple_val_paths(parquet_module_path, tensor_schema, max_len, batch_size=4):
    transforms = make_default_sasrec_transforms(tensor_schema, query_column="user_id")

    metadata = {
        "train": create_shared_meta(shape=max_len + 1),
        "validate": create_shared_meta(shape=max_len),
        "test": create_shared_meta(shape=max_len),
        "predict": create_shared_meta(shape=max_len),
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
>>>>>>> Add saving/loading `linucb_arms` in pickle format
