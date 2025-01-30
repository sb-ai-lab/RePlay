import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.callbacks import PandasPredictionCallback
    from replay.models.nn.sequential.sasrec import (
        OptimizedSasRec,
        SasRec,
        SasRecPredictionBatch,
        SasRecPredictionDataset,
    )


torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "use_candidates, candidates",
    [
        (True, torch.LongTensor([1])),
        (True, torch.LongTensor([1, 2, 3, 4])),
        (True, torch.LongTensor([0, 1, 2, 3, 4, 5])),
        (False, None),
    ],
)
@pytest.mark.parametrize(
    "mode, batch_size",
    [
        ("one_query", 1),
        ("batch", 2),
        ("batch", 3),
        ("dynamic_batch_size", 2),
        ("dynamic_batch_size", 3),
        ("dynamic_one_query", 1),
    ],
)
def test_prediction_optimized_sasrec(
    item_user_sequential_dataset, train_sasrec_loader, use_candidates, candidates, mode, batch_size
):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred, batch_size=batch_size)
    cardinality = item_user_sequential_dataset.schema._get_object_args()[0]["cardinality"]
    num_preds = len(pred) if mode != "batch" else len(pred) // batch_size * batch_size

    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    pandas_prediction_callback = PandasPredictionCallback(
        top_k=candidates.shape[0] if candidates is not None else cardinality,
        query_column="user_id",
        item_column="item_id",
        rating_column="score",
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_sasrec_loader)
    trainer.save_checkpoint("test.ckpt")

    opt_model = OptimizedSasRec.from_checkpoint(
        checkpoint_path="test.ckpt",
        mode=mode,
        batch_size=batch_size if mode != "dynamic_batch_size" else None,
        use_candidates_to_score=use_candidates,
    )

    if candidates is not None:
        opt_model.predict_dataloader(
            dataloader=pred_sasrec_loader, callbacks=[pandas_prediction_callback], candidates_to_score=candidates
        )
        assert pandas_prediction_callback.get_result()["score"].shape[0] == num_preds * candidates.shape[0]
    else:
        opt_model.predict_dataloader(dataloader=pred_sasrec_loader, callbacks=[pandas_prediction_callback])
        assert pandas_prediction_callback.get_result()["score"].shape[0] == num_preds * cardinality


@pytest.mark.torch
@pytest.mark.parametrize(
    "use_candidates, candidates",
    [
        (True, torch.FloatTensor([1])),
        (True, torch.LongTensor([1]).numpy()),
        (False, torch.LongTensor([1])),
    ],
)
def test_prediction_optimized_sasrec_invalid_candidates_to_score(
    item_user_sequential_dataset, train_sasrec_loader, use_candidates, candidates
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
            model = OptimizedSasRec.from_checkpoint(
                checkpoint_path="test.ckpt",
                mode="one_query",
                use_candidates_to_score=use_candidates,
            )
            model.predict(batch, candidates)


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

    opt_model = OptimizedSasRec.from_checkpoint(checkpoint_path="test.ckpt")

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
        OptimizedSasRec.from_checkpoint(checkpoint_path="csome_path", mode="invalid_mode")
