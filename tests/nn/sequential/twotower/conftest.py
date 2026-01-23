import pytest

pytest.importorskip("torch")
import torch

from replay.data.nn.schema import TensorMap
from replay.nn.agg import ConcatAggregator
from replay.nn.embedding import SequenceEmbedding
from replay.nn.ffn import SwiGLUEncoder
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.mask import DefaultAttentionMask
from replay.nn.sequential import DiffTransformerLayer, PositionAwareAggregator, TwoTower, TwoTowerBody


@pytest.fixture(
    params=[
        (CE, {"padding_idx": 40}),
        (CESampled, {"padding_idx": 40}),
        (BCE, {}),
        (BCESampled, {}),
        (LogOutCE, {"padding_idx": 40, "vocab_size": 40}),
        (LogInCE, {"vocab_size": 40}),
        (LogInCESampled, {}),
    ],
    ids=["CE", "CE sampled", "BCE", "BCE sampled", "LogOutCE", "LogInCE", "LogInCESampled"],
)
def twotower_parametrized(request, tensor_schema, item_features_path):
    loss_cls, kwargs = request.param
    loss = loss_cls(**kwargs)

    common_aggregator = ConcatAggregator(
        input_embedding_dims=[64, 64, 64, 64, 64],
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
def twotower_model(tensor_schema, item_features_path):
    model = TwoTower.from_params(
        schema=tensor_schema,
        item_features_path=item_features_path,
        embedding_dim=64,
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
