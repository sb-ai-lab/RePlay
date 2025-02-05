import pathlib

import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.sasrec import (
        SasRec,
        SasRecPredictionBatch,
        SasRecPredictionDataset,
    )
    from replay.models.nn.sequential.sasrec.optimized_model import OptimizedSasRec


torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "num_candidates, candidates",
    [
        (1, torch.LongTensor([1])),
        (4, torch.LongTensor([1, 2, 3, 4])),
        (6, torch.LongTensor([0, 1, 2, 3, 4, 5])),
        (None, None),
        (-1, torch.LongTensor([1])),
        (-1, torch.LongTensor([1, 2, 3, 4])),
        (-1, torch.LongTensor([0, 1, 2, 3, 4, 5])),
    ],
)
@pytest.mark.parametrize(
    "mode, batch_size",
    [
        ("one_query", 1),
        ("batch", 2),
        ("dynamic_batch_size", 2),
        ("dynamic_batch_size", 3),
    ],
)
def test_prediction_optimized_sasrec(
    item_user_sequential_dataset, train_sasrec_loader, num_candidates, candidates, mode, batch_size
):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred, batch_size=batch_size)
    cardinality = item_user_sequential_dataset.schema._get_object_args()[0]["cardinality"]

    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_sasrec_loader)
    trainer.save_checkpoint("test.ckpt")

    opt_model = OptimizedSasRec.compile(
        model="test.ckpt",
        mode=mode,
        batch_size=batch_size if mode != "dynamic_batch_size" else None,
        num_candidates_to_score=num_candidates,
    )

    if candidates is not None:
        for batch in pred_sasrec_loader:
            scores = opt_model.predict(batch=batch, candidates_to_score=candidates)
            assert scores.shape == (batch.padding_mask.shape[0], candidates.shape[0])
    else:
        for batch in pred_sasrec_loader:
            scores = opt_model.predict(batch=batch)
            assert scores.shape == (batch.padding_mask.shape[0], cardinality)


@pytest.mark.torch
@pytest.mark.parametrize(
    "num_candidates, candidates",
    [
        (1, torch.FloatTensor([1])),
        (1, torch.LongTensor([1]).numpy()),
        (None, torch.LongTensor([1])),
        (-10, torch.LongTensor([1])),
        (1.5, torch.LongTensor([1])),
    ],
)
def test_prediction_optimized_sasrec_invalid_candidates_to_score(
    item_user_sequential_dataset, train_sasrec_loader, num_candidates, candidates
):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred)
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_sasrec_loader)
    trainer.save_checkpoint("test.ckpt")

    for i, batch in enumerate(pred_sasrec_loader):
        with pytest.raises(ValueError):
            model = OptimizedSasRec.compile(
                model="test.ckpt",
                mode="one_query",
                num_candidates_to_score=num_candidates,
            )
            model.predict(batch, candidates)


@pytest.mark.torch
@pytest.mark.parametrize(
    "mode, batch_size",
    [
        ("batch", 3),
    ],
)
def test_prediction_optimized_sasrec_invalid_batch_in_batch_mode(
    item_user_sequential_dataset, train_sasrec_loader, mode, batch_size
):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred, batch_size=batch_size)

    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_sasrec_loader)
    trainer.save_checkpoint("test.ckpt")

    opt_model = OptimizedSasRec.compile(
        model="test.ckpt",
        mode=mode,
        batch_size=batch_size * 100,
    )

    with pytest.raises(ValueError):
        for batch in pred_sasrec_loader:
            opt_model.predict(batch=batch)


@pytest.mark.torch
def test_optimized_sasrec_prepare_prediction_batch(item_user_sequential_dataset, train_sasrec_loader):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred, batch_size=1)

    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_sasrec_loader)
    trainer.save_checkpoint("test.ckpt")

    opt_model = OptimizedSasRec.compile(model="test.ckpt")

    for i, batch in enumerate(pred_sasrec_loader):
        query_id, padding_mask, features = batch
        padding_mask_double = torch.cat((padding_mask, padding_mask), 1)
        padding_mask_reduced = padding_mask[:, :-2]
        batch1 = SasRecPredictionBatch(query_id, padding_mask_double, features)
        batch2 = SasRecPredictionBatch(query_id, padding_mask_reduced, features)
        break

    with pytest.raises(ValueError):
        opt_model._prepare_prediction_batch(batch1)

    new_batch = opt_model._prepare_prediction_batch(batch2)
    assert new_batch.padding_mask.shape[1] == opt_model._max_seq_len


@pytest.mark.torch
def test_optimized_sasrec_invalid_mode():
    with pytest.raises(ValueError):
        OptimizedSasRec.compile(model="some_path", mode="invalid_mode")


@pytest.mark.torch
def test_optimized_sasrec_compile_from_different_sources(item_user_sequential_dataset, train_sasrec_loader):
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_sasrec_loader)
    trainer.save_checkpoint("test.ckpt")

    opt_model1 = OptimizedSasRec.compile(model="test.ckpt")
    opt_model2 = OptimizedSasRec.compile(model=pathlib.Path("test.ckpt"))
    opt_model3 = OptimizedSasRec.compile(model=model)
    assert str(opt_model1._model) == str(opt_model2._model) == str(opt_model3._model)
