import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.callbacks import PandasPredictionCallback
    from replay.models.nn.sequential.sasrec import (
        OptimizedSasRec,
        SasRec,
        SasRecPredictionDataset,
    )


torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "number, candidates",
    [
        (1, torch.LongTensor([1])),
        (4, torch.LongTensor([1, 2, 3, 4])),
        (6, torch.LongTensor([0, 1, 2, 3, 4, 5])),
        (None, None),
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
def test_prediction_optimized_sasrec_fixed_axis_dataloader(
    item_user_sequential_dataset, train_sasrec_loader, number, candidates, mode, batch_size
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
        top_k=number if number is not None else cardinality,
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
        num_candidates_to_score=number,
    )

    if number is not None:
        opt_model.predict_dataloader(
            dataloader=pred_sasrec_loader, callbacks=[pandas_prediction_callback], candidates_to_score=candidates
        )
        assert pandas_prediction_callback.get_result()["score"].shape[0] == num_preds * number
    else:
        opt_model.predict_dataloader(dataloader=pred_sasrec_loader, callbacks=[pandas_prediction_callback])
        assert pandas_prediction_callback.get_result()["score"].shape[0] == num_preds * cardinality


@pytest.mark.torch
@pytest.mark.parametrize(
    "number, candidates",
    [
        (-1, torch.LongTensor([1])),
        (0, torch.LongTensor([1])),
        (1.5, torch.LongTensor([1])),
        (1.5, torch.FloatTensor([1])),
        (None, torch.LongTensor([1])),
    ],
)
def test_prediction_optimized_sasrec_invalid_candidates_to_score(
    item_user_sequential_dataset, train_sasrec_loader, number, candidates
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
                num_candidates_to_score=number,
            )
            model.predict(batch, candidates)
