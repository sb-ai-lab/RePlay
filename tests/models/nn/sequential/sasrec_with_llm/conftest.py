import pandas as pd
import pytest

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import lightning as L
    import torch

    from replay.data.nn import SequenceTokenizer, TensorFeatureInfo, TensorFeatureSource, TensorSchema
    from replay.models.nn.sequential.sasrec_with_llm import SasRecLLM, SasRecLLMTrainingDataset


@pytest.fixture(scope="module")
def feature_schema_for_sasrec():
    schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_id",
                feature_hint=FeatureHint.QUERY_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureInfo(
                column="item_id",
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )

    return schema


@pytest.fixture(scope="class")
def fitted_sasrec(feature_schema_for_sasrec):
    data = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3, 1, 2],
        }
    )

    profile_emb_dim = 1024

    user_profile_embeddings = torch.zeros((3, profile_emb_dim), dtype=torch.float32)
    existing_profile_binary_mask = torch.BoolTensor([True, True, True])

    train_dataset = Dataset(feature_schema=feature_schema_for_sasrec, interactions=data)
    tensor_schema = TensorSchema(
        TensorFeatureInfo(
            name="item_id",
            is_seq=True,
            cardinality=train_dataset.item_count,
            feature_type=FeatureType.CATEGORICAL,
            feature_sources=[
                TensorFeatureSource(FeatureSource.INTERACTIONS, train_dataset.feature_schema.item_id_column)
            ],
            feature_hint=FeatureHint.ITEM_ID,
        )
    )

    tokenizer = SequenceTokenizer(tensor_schema, allow_collect_to_master=True)
    tokenizer.fit(train_dataset)
    sequential_train_dataset = tokenizer.transform(train_dataset)

    model = SasRecLLM(tensor_schema, profile_emb_dim=profile_emb_dim, max_seq_len=5)
    trainer = L.Trainer(max_epochs=1)
    train_loader = torch.utils.data.DataLoader(
        SasRecLLMTrainingDataset(
            sequential=sequential_train_dataset,
            max_sequence_length=5,
            user_profile_embeddings=user_profile_embeddings,
            existing_profile_binary_mask=existing_profile_binary_mask,
        )
    )

    trainer.fit(model, train_dataloaders=train_loader)

    return model, tokenizer


@pytest.fixture(scope="module")
def new_items_dataset():
    data = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3, 1, 2, 4],
        }
    )

    return data
