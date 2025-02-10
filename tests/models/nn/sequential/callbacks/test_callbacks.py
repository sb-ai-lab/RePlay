import pytest

from replay.utils import (
    PYSPARK_AVAILABLE,
    TORCH_AVAILABLE,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
    get_spark_session,
)

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.bert4rec import Bert4Rec, Bert4RecPredictionDataset
    from replay.models.nn.sequential.callbacks import (
        PandasPredictionCallback,
        PolarsPredictionCallback,
        QueryEmbeddingsPredictionCallback,
        TorchPredictionCallback,
        ValidationMetricsCallback,
    )
    from replay.models.nn.sequential.postprocessors import RemoveSeenItems
    from replay.models.nn.sequential.sasrec import SasRec, SasRecPredictionDataset

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
@pytest.mark.parametrize(
    "candidates", [torch.LongTensor([0]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3, 4]), None]
)
@pytest.mark.parametrize(
    "model, dataset, train_dataloader",
    [
        (Bert4Rec, Bert4RecPredictionDataset, "train_loader"),
        (SasRec, SasRecPredictionDataset, "train_sasrec_loader"),
    ],
)
def test_torch_prediction_callback_fast_forward(
    item_user_sequential_dataset, is_postprocessor, candidates, model, dataset, train_dataloader, request
):
    cardinality = item_user_sequential_dataset.schema["item_id"].cardinality
    pred = dataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    callback = TorchPredictionCallback(
        1,
        postprocessors=(
            [RemoveSeenItems(item_user_sequential_dataset, candidates=candidates if candidates is not None else None)]
            if is_postprocessor
            else None
        ),
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = model(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, request.getfixturevalue(train_dataloader))
    if candidates is not None:
        if isinstance(model, SasRec):
            model.candidates_to_score = candidates
        else:
            with pytest.raises(NotImplementedError):
                model.candidates_to_score = candidates
            return
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    if candidates is None:
        assert predicted[0].size() == (1, cardinality)
    else:
        assert predicted[0].size() == (1, candidates.shape[0])
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
@pytest.mark.parametrize(
    "candidates", [torch.LongTensor([0]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3, 4]), None]
)
@pytest.mark.parametrize(
    "model, dataset, train_dataloader",
    [
        (Bert4Rec, Bert4RecPredictionDataset, "train_loader"),
        (SasRec, SasRecPredictionDataset, "train_sasrec_loader"),
    ],
)
def test_pandas_prediction_callback_fast_forward(
    item_user_sequential_dataset, is_postprocessor, candidates, model, dataset, train_dataloader, request
):
    cardinality = item_user_sequential_dataset.schema["item_id"].cardinality
    pred = dataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    callback = PandasPredictionCallback(
        1,
        "user_id",
        "item_id",
        postprocessors=(
            [RemoveSeenItems(item_user_sequential_dataset, candidates=candidates if candidates is not None else None)]
            if is_postprocessor
            else None
        ),
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = model(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, request.getfixturevalue(train_dataloader))
    if candidates is not None:
        if isinstance(model, SasRec):
            model.candidates_to_score = candidates
        else:
            with pytest.raises(NotImplementedError):
                model.candidates_to_score = candidates
            return
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    if candidates is None:
        assert predicted[0].size() == (1, cardinality)
    else:
        assert predicted[0].size() == (1, candidates.shape[0])
    assert isinstance(callback.get_result(), PandasDataFrame)


@pytest.mark.torch
@pytest.mark.parametrize(
    "is_postprocessor",
    [
        (False),
        (True),
    ],
)
@pytest.mark.parametrize(
    "candidates", [torch.LongTensor([0]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3, 4]), None]
)
@pytest.mark.parametrize(
    "model, dataset, train_dataloader",
    [
        (Bert4Rec, Bert4RecPredictionDataset, "train_loader"),
        (SasRec, SasRecPredictionDataset, "train_sasrec_loader"),
    ],
)
def test_polars_prediction_callback_fast_forward(
    item_user_sequential_dataset, is_postprocessor, candidates, model, dataset, train_dataloader, request
):
    cardinality = item_user_sequential_dataset.schema["item_id"].cardinality
    pred = dataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    callback = PolarsPredictionCallback(
        1,
        "user_id",
        "item_id",
        postprocessors=(
            [RemoveSeenItems(item_user_sequential_dataset, candidates=candidates if candidates is not None else None)]
            if is_postprocessor
            else None
        ),
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = model(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, request.getfixturevalue(train_dataloader))
    if candidates is not None:
        if isinstance(model, SasRec):
            model.candidates_to_score = candidates
        else:
            with pytest.raises(NotImplementedError):
                model.candidates_to_score = candidates
            return
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    if candidates is None:
        assert predicted[0].size() == (1, cardinality)
    else:
        assert predicted[0].size() == (1, candidates.shape[0])
    assert isinstance(callback.get_result(), PolarsDataFrame)


@pytest.mark.torch
@pytest.mark.spark
@pytest.mark.parametrize(
    "is_postprocessor",
    [
        (False),
        (True),
    ],
)
@pytest.mark.parametrize(
    "candidates", [torch.LongTensor([0]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3, 4]), None]
)
@pytest.mark.parametrize(
    "model, dataset, train_dataloader",
    [
        (Bert4Rec, Bert4RecPredictionDataset, "train_loader"),
        (SasRec, SasRecPredictionDataset, "train_sasrec_loader"),
    ],
)
def test_spark_prediction_callback_fast_forward(
    item_user_sequential_dataset, is_postprocessor, candidates, model, dataset, train_dataloader, request
):
    cardinality = item_user_sequential_dataset.schema["item_id"].cardinality
    pred = dataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    callback = SparkPredictionCallback(
        1,
        "user_id",
        "item_id",
        "score",
        get_spark_session(),
        postprocessors=(
            [RemoveSeenItems(item_user_sequential_dataset, candidates=candidates if candidates is not None else None)]
            if is_postprocessor
            else None
        ),
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = model(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, request.getfixturevalue(train_dataloader))
    if candidates is not None:
        if isinstance(model, SasRec):
            model.candidates_to_score = candidates
        else:
            with pytest.raises(NotImplementedError):
                model.candidates_to_score = candidates
            return
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    if candidates is None:
        assert predicted[0].size() == (1, cardinality)
    else:
        assert predicted[0].size() == (1, candidates.shape[0])
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
@pytest.mark.parametrize(
    "candidates", [torch.LongTensor([0]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3, 4]), None]
)
@pytest.mark.parametrize(
    "model, dataset, train_dataloader, val_dataloader",
    [
        (Bert4Rec, Bert4RecPredictionDataset, "train_loader", "val_loader"),
        (SasRec, SasRecPredictionDataset, "train_sasrec_loader", "val_sasrec_loader"),
    ],
)
def test_validation_callbacks(
    item_user_sequential_dataset,
    metrics,
    postprocessor,
    candidates,
    model,
    dataset,
    train_dataloader,
    val_dataloader,
    request,
):
    cardinality = item_user_sequential_dataset.schema["item_id"].cardinality
    callback = ValidationMetricsCallback(
        metrics=metrics,
        ks=[1],
        item_count=1,
        postprocessors=(
            [postprocessor(item_user_sequential_dataset, candidates=candidates if candidates is not None else None)]
            if postprocessor
            else None
        ),
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = model(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
    )
    trainer.fit(model, request.getfixturevalue(train_dataloader), request.getfixturevalue(val_dataloader))
    if candidates is not None:
        if isinstance(model, SasRec):
            model.candidates_to_score = candidates
        else:
            with pytest.raises(NotImplementedError):
                model.candidates_to_score = candidates
            return

    pred = dataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    if candidates is None:
        assert predicted[0].size() == (1, cardinality)
    else:
        assert predicted[0].size() == (1, candidates.shape[0])


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
@pytest.mark.parametrize(
    "candidates", [torch.LongTensor([0]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3, 4]), None]
)
@pytest.mark.parametrize(
    "model, dataset, train_dataloader, val_dataloader",
    [
        (Bert4Rec, Bert4RecPredictionDataset, "train_loader", "val_loader"),
        (SasRec, SasRecPredictionDataset, "train_sasrec_loader", "val_sasrec_loader"),
    ],
)
def test_validation_callbacks_multiple_dataloaders(
    item_user_sequential_dataset,
    metrics,
    postprocessor,
    candidates,
    model,
    dataset,
    train_dataloader,
    val_dataloader,
    request,
):
    cardinality = item_user_sequential_dataset.schema["item_id"].cardinality
    callback = ValidationMetricsCallback(
        metrics=metrics,
        ks=[1],
        item_count=1,
        postprocessors=(
            [postprocessor(item_user_sequential_dataset, candidates=candidates if candidates is not None else None)]
            if postprocessor
            else None
        ),
    )

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    model = model(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
    )
    val_loader = request.getfixturevalue(val_dataloader)
    trainer.fit(model, request.getfixturevalue(train_dataloader), [val_loader, val_loader])

    if candidates is not None:
        if isinstance(model, SasRec):
            model.candidates_to_score = candidates
        else:
            with pytest.raises(NotImplementedError):
                model.candidates_to_score = candidates
            return

    pred = dataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    if candidates is None:
        assert predicted[0].size() == (1, cardinality)
    else:
        assert predicted[0].size() == (1, candidates.shape[0])


@pytest.mark.torch
@pytest.mark.parametrize(
    "model, dataset",
    [
        (Bert4Rec, Bert4RecPredictionDataset),
        (SasRec, SasRecPredictionDataset),
    ],
)
@pytest.mark.parametrize(
    "candidates", [torch.LongTensor([0]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3, 4]), None]
)
def test_query_embeddings_callback(item_user_sequential_dataset, candidates, model, dataset):
    callback = QueryEmbeddingsPredictionCallback()
    model = model(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
    )
    if candidates is not None:
        if isinstance(model, SasRec):
            model.candidates_to_score = candidates
        else:
            with pytest.raises(NotImplementedError):
                model.candidates_to_score = candidates
            return

    pred = dataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)

    trainer = L.Trainer(callbacks=[callback], inference_mode=True)
    trainer.predict(model, pred_loader)
    embs = callback.get_result()

    assert embs.shape == (4, 64)
