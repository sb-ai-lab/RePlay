import pytest
import torch

from replay.nn import ConcatAggregator, DefaultAttentionMask, SequenceEmbedding
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn.sequential.sasrec import DiffTransformerLayer, PositionAwareAggregator, SasRec, SasRecBody


@pytest.fixture(
    params=[
        (CE, {"padding_idx": 40}),
        (CESampled, {"padding_idx": 40}),
        (BCE, {}),
        (BCESampled, {}),
        (LogOutCE, {"padding_idx": 40, "vocab_size": 40}),
        (LogInCE, {"vocab_size": 41}),
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
