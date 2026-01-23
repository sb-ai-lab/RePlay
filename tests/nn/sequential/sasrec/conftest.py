import pytest
import torch

from replay.data.nn import ParquetModule
from replay.nn.agg import ConcatAggregator
from replay.nn.embedding import SequenceEmbedding
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.mask import DefaultAttentionMask
from replay.nn.sequential import DiffTransformerLayer, PositionAwareAggregator, SasRec, SasRecBody
from replay.nn.transform.template import make_default_sasrec_transforms


@pytest.fixture(
    params=[
        (CE, {"padding_idx": 40}),
        (CESampled, {"padding_idx": 40}),
        (BCE, {}),
        (BCESampled, {}),
        (LogOutCE, {"padding_idx": 40, "vocab_size": 40}),
        (LogInCE, {"vocab_size": 40}),  # 41?
        (LogInCESampled, {}),
    ],
    ids=["CE", "CE sampled", "BCE", "BCE sampled", "LogOutCE", "LogInCE", "LogInCESampled"],
)
def sasrec_parametrized(request, tensor_schema):
    loss_cls, kwargs = request.param
    loss = loss_cls(**kwargs)

    body = SasRecBody(
        embedder=SequenceEmbedding(tensor_schema, categorical_list_feature_aggregation_method="sum"),
        embedding_aggregator=(
            PositionAwareAggregator(
                ConcatAggregator(
                    input_embedding_dims=[64, 64, 64, 64, 64],
                    output_embedding_dim=64,
                ),
                max_sequence_length=7,
                dropout=0.2,
            )
        ),
        attn_mask_builder=DefaultAttentionMask("item_id", 1),
        encoder=DiffTransformerLayer(embedding_dim=64, num_heads=1, num_blocks=1),
        output_normalization=torch.nn.LayerNorm(64),
    )
    model = SasRec(body=body, loss=loss)
    return model


@pytest.fixture
def sasrec_model_only_items(tensor_schema):
    model = SasRec.from_params(
        schema=tensor_schema.filter(name="item_id"),
        embedding_dim=64,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


@pytest.fixture
def parquet_module_with_default_sasrec_transform(parquet_module_path, tensor_schema, max_len, batch_size=4):
    transforms = make_default_sasrec_transforms(tensor_schema, query_column="user_id")

    def create_meta(shape):
        shared_meta = {
            "user_id": {},
            "item_id": {"shape": shape, "padding": tensor_schema["item_id"].padding_value},
        }
        return shared_meta

    metadata = {
        "train": create_meta(shape=max_len + 1),
        "validate": create_meta(shape=max_len),
        "test": create_meta(shape=max_len),
        "predict": create_meta(shape=max_len),
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
