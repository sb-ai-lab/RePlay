import pytest

pytest.importorskip("torch")
import torch

from replay.data.nn import ParquetModule
from replay.data.nn.schema import TensorMap
from replay.nn.agg import ConcatAggregator
from replay.nn.embedding import SequenceEmbedding
from replay.nn.ffn import SwiGLUEncoder
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.mask import DefaultAttentionMask
from replay.nn.sequential import DiffTransformerLayer, PositionAwareAggregator, TwoTower, TwoTowerBody
from replay.nn.transform.template import make_default_twotower_transforms


@pytest.fixture(
    params=[
        (CE, {"ignore_index": 40}),
        (CESampled, {"ignore_index": 40}),
        (BCE, {}),
        (BCESampled, {}),
        (LogOutCE, {"ignore_index": 40, "cardinality": 40}),
        (LogInCE, {"cardinality": 40}),
        (LogInCESampled, {}),
    ],
    ids=["CE", "CE sampled", "BCE", "BCE sampled", "LogOutCE", "LogInCE", "LogInCESampled"],
)
def twotower_parametrized(request, tensor_schema, item_features_path):
    loss_cls, kwargs = request.param
    loss = loss_cls(**kwargs)

    common_aggregator = ConcatAggregator(
        input_embedding_dims=[x.embedding_dim for x in tensor_schema.values()],
        output_embedding_dim=64,
    )

    body = TwoTowerBody(
        schema=tensor_schema,
        embedder=SequenceEmbedding(
            schema=tensor_schema,
            categorical_list_feature_aggregation_method="sum",
        ),
        attn_mask_builder=DefaultAttentionMask(
            reference_feature_name=tensor_schema.item_id_feature_name,
            num_heads=1,
        ),
        query_tower_feature_names=tensor_schema.names,
        item_tower_feature_names=tensor_schema.names,
        query_embedding_aggregator=PositionAwareAggregator(
            embedding_aggregator=common_aggregator,
            max_sequence_length=7,
            dropout=0.2,
        ),
        item_embedding_aggregator=common_aggregator,
        query_encoder=DiffTransformerLayer(
            embedding_dim=64,
            num_heads=1,
            num_blocks=1,
        ),
        query_tower_output_normalization=torch.nn.LayerNorm(64),
        item_encoder=SwiGLUEncoder(embedding_dim=64, hidden_dim=2 * 64),
        item_features_path=item_features_path,
    )
    model = TwoTower(
        body=body,
        loss=loss,
        context_merger=None,
    )

    return model


@pytest.fixture
def twotower_model(tensor_schema_with_equal_embedding_dims, item_features_path):
    model = TwoTower.from_params(
        schema=tensor_schema_with_equal_embedding_dims,
        item_features_path=item_features_path,
        embedding_dim=70,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


class DummyContextMerger(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        model_hidden_state: torch.Tensor,
        feature_tensors: TensorMap,
    ) -> torch.Tensor:
        return model_hidden_state


@pytest.fixture
def twotower_model_with_context_merger(twotower_model):
    twotower_model.context_merger = DummyContextMerger()
    return twotower_model


@pytest.fixture
def twotower_model_only_items(tensor_schema_with_equal_embedding_dims, item_features_path):
    model = TwoTower.from_params(
        schema=tensor_schema_with_equal_embedding_dims.filter(name="item_id"),
        item_features_path=item_features_path,
        embedding_dim=70,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model


@pytest.fixture
def parquet_module_with_default_twotower_transform(
    parquet_module_path, tensor_schema_with_equal_embedding_dims, max_len, batch_size=4
):
    transforms = make_default_twotower_transforms(tensor_schema_with_equal_embedding_dims, query_column="user_id")

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
