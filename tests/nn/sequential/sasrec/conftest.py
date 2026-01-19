import copy

import pytest
import torch

from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE
from replay.nn import ConcatAggregator, SequenceEmbedding, DefaultAttentionMask

from replay.nn.sequential.sasrec import DiffTransformerLayer, SasRecAggregator, SasRec


# @pytest.fixture(scope="module")
# def sasrec_training_dataset(sequential_dataset, tensor_schema):
#     dataset = SasRecTrainingDataset(
#         sequential_dataset,
#         negative_sampler=SequentialNegativeSampler(
#             vocab_size=tensor_schema["item_id"].cardinality,
#             item_id_feature_name="item_id",
#             negative_sampling_strategy="global_uniform",
#             num_negative_samples=2,
#         ),
#         max_sequence_length=7,
#     )
#     return dataset


# @pytest.fixture(scope="module")
# def sequential_sample(sasrec_training_dataset):
#     return sasrec_training_dataset.collate_fn([sasrec_training_dataset[0], sasrec_training_dataset[1]])


# @pytest.fixture(scope="module")
# def wrong_sequential_sample(request, sequential_sample):
#     sample = copy.deepcopy(sequential_sample)

#     defect_type = request.param
#     if defect_type == "missing field":
#         sample.pop("padding_mask")
#     elif defect_type == "wrong length":
#         sample["feature_tensors"]["item_id"] = sample["feature_tensors"]["item_id"][:, 1:]
#     elif defect_type == "index out of embedding":
#         sample["feature_tensors"]["item_id"][0][-1] = 4
#     else:
#         raise ValueError(defect_type)
#     return sample


# @pytest.fixture(scope="module")
# def sasrec_train_dataloader(sasrec_training_dataset):
#     train_dataloader = torch.utils.data.DataLoader(
#         sasrec_training_dataset,
#         batch_size=2,
#         collate_fn=getattr(sasrec_training_dataset, "collate_fn", None),
#     )
#     return train_dataloader


# @pytest.fixture(scope="module")
# def wrong_builder(tensor_schema):
#     builder = SasRecBuilder().default(tensor_schema)
#     builder = builder.loss(None)
#     return builder


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

    model = SasRec(
        embedder=SequenceEmbedding(
                tensor_schema,
                categorical_list_feature_aggregation_method="sum"
        ),
        embedding_aggregator=(
            SasRecAggregator(
                ConcatAggregator(
                                input_embedding_dims=[64, 64, 64, 64, 64],
                                # input_embedding_dims=[64, 64],
                                output_embedding_dim=64),
                max_sequence_length=7,
                dropout=0.2,
            )
        ),
        attn_mask_builder=DefaultAttentionMask("item_id", 1),
        encoder=DiffTransformerLayer(embedding_dim=64, num_heads=1, num_blocks=1),
        output_normalization=torch.nn.LayerNorm(64),
        loss=loss
    )
    return model
