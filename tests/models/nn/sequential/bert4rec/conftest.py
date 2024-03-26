import pandas as pd
import pytest

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import lightning as L
    import torch

    from replay.data.nn import SequenceTokenizer, TensorFeatureInfo, TensorFeatureSource, TensorSchema
    from replay.models.nn.sequential.bert4rec import Bert4Rec, Bert4RecTrainingDataset


@pytest.fixture()
def feature_schema_for_bert4rec():
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


@pytest.fixture()
def fitted_bert4rec(feature_schema_for_bert4rec):
    data = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3, 1, 2],
        }
    )

    train_dataset = Dataset(feature_schema=feature_schema_for_bert4rec, interactions=data)
    tensor_schema = TensorSchema(
        TensorFeatureInfo(
            name="item_id_seq",
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

    model = Bert4Rec(tensor_schema, max_seq_len=5, hidden_size=64)
    trainer = L.Trainer(max_epochs=1)
    train_loader = torch.utils.data.DataLoader(Bert4RecTrainingDataset(sequential_train_dataset, 5))

    trainer.fit(model, train_dataloaders=train_loader)

    return model, tokenizer


@pytest.fixture()
def fitted_bert4rec_enable_embedding_tying(feature_schema_for_bert4rec):
    data = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3, 1, 2],
        }
    )

    train_dataset = Dataset(feature_schema=feature_schema_for_bert4rec, interactions=data)
    tensor_schema = TensorSchema(
        TensorFeatureInfo(
            name="item_id_seq",
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

    model = Bert4Rec(tensor_schema, max_seq_len=5, hidden_size=64, enable_embedding_tying=True)
    trainer = L.Trainer(max_epochs=1)
    train_loader = torch.utils.data.DataLoader(Bert4RecTrainingDataset(sequential_train_dataset, 5))

    trainer.fit(model, train_dataloaders=train_loader)

    return model, tokenizer


@pytest.fixture()
def new_items_dataset():
    data = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3, 1, 2, 4],
        }
    )

    return data
