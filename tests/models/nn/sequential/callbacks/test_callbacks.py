import pytest

from replay.utils import PYSPARK_AVAILABLE, TORCH_AVAILABLE, PandasDataFrame, SparkDataFrame, get_spark_session

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.bert4rec import Bert4Rec, Bert4RecPredictionDataset
    from replay.models.nn.sequential.callbacks import (
        PandasPredictionCallback,
        TorchPredictionCallback,
        ValidationMetricsCallback,
        QueryEmbeddingsPredictionCallback,
    )
    from replay.models.nn.sequential.postprocessors import RemoveSeenItems

    if PYSPARK_AVAILABLE:
        from replay.models.nn.sequential.callbacks import SparkPredictionCallback

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "is_postprocessor",
    [
        (False),
        (True),
    ],
)
def test_torch_prediction_callback_fast_forward(item_user_sequential_dataset, train_loader, is_postprocessor):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    callback = TorchPredictionCallback(
        1,
        postprocessors=[RemoveSeenItems(item_user_sequential_dataset)] if is_postprocessor else None,
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_loader)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)
    users, items, scores = callback.get_result()
    assert isinstance(users, torch.LongTensor)
    assert isinstance(items, torch.LongTensor)
    assert isinstance(scores, torch.Tensor)


@pytest.mark.torch
@pytest.mark.parametrize(
    "is_postprocessor",
    [
        (False),
        (True),
    ],
)
def test_pandas_prediction_callback_fast_forward(item_user_sequential_dataset, train_loader, is_postprocessor):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    callback = PandasPredictionCallback(
        1,
        "user_id",
        "item_id",
        postprocessors=[RemoveSeenItems(item_user_sequential_dataset)] if is_postprocessor else None,
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_loader)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)
    assert isinstance(callback.get_result(), PandasDataFrame)


@pytest.mark.torch
@pytest.mark.spark
@pytest.mark.parametrize(
    "is_postprocessor",
    [
        (False),
        (True),
    ],
)
def test_spark_prediction_callback_fast_forward(item_user_sequential_dataset, train_loader, is_postprocessor):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    callback = SparkPredictionCallback(
        1,
        "user_id",
        "item_id",
        "score",
        get_spark_session(),
        postprocessors=[RemoveSeenItems(item_user_sequential_dataset)] if is_postprocessor else None,
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_loader)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)
    assert isinstance(callback.get_result(), SparkDataFrame)


@pytest.mark.torch
@pytest.mark.parametrize(
    "metrics, postprocessor",
    [
        (["coverage", "precision"], RemoveSeenItems),
        (["coverage"], RemoveSeenItems),
        (["coverage", "precision"], None),
        (["coverage"], None),
    ],
)
def test_validation_callbacks(item_user_sequential_dataset, train_loader, val_loader, metrics, postprocessor):
    callback = ValidationMetricsCallback(
        metrics=metrics,
        ks=[1],
        item_count=1,
        postprocessors=[postprocessor(item_user_sequential_dataset)] if postprocessor else None,
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
    )
    trainer.fit(model, train_loader, val_loader)

    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)


@pytest.mark.torch
@pytest.mark.parametrize(
    "metrics, postprocessor",
    [
        (["coverage", "precision"], RemoveSeenItems),
        (["coverage"], RemoveSeenItems),
        (["coverage", "precision"], None),
        (["coverage"], None),
    ],
)
def test_validation_callbacks_multiple_dataloaders(item_user_sequential_dataset, train_loader, val_loader, metrics, postprocessor):
    callback = ValidationMetricsCallback(
        metrics=metrics,
        ks=[1],
        item_count=1,
        postprocessors=[postprocessor(item_user_sequential_dataset)] if postprocessor else None,
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
    )
    trainer.fit(model, train_loader, [val_loader, val_loader])

    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)


@pytest.mark.torch
def test_query_embeddings_callback(item_user_sequential_dataset):
    callback = QueryEmbeddingsPredictionCallback()
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
    )
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    trainer = L.Trainer(
        callbacks=[callback],
        inference_mode=True
    )
    trainer.predict(model, pred_loader)
    embs = callback.get_result()

    assert embs.shape == (4, 64)
