import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.sasrec import (
        OptimizedSasRec,
        SasRec,
        SasRecPredictionDataset,
    )


torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_dict",
    [
        {"number": 3, "candidates": torch.LongTensor([1, 2, 3]), "output_size": 3},
        {"number": None, "candidates": None, "output_size": 6},
    ],
)
def test_prediction_optimized_sasrec(item_user_sequential_dataset, train_sasrec_loader, candidates_dict):
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

    opt_model = OptimizedSasRec.from_checkpoint(
        checkpoint_path="test.ckpt",
        mode="one_query",
        num_candidates_to_score=candidates_dict["number"],
    )
    for i, batch in enumerate(pred_sasrec_loader):
        if candidates_dict["number"] is None:
            scores = opt_model.predict(batch)
        else:
            scores = opt_model.predict(batch, candidates_dict["candidates"])
        break
    assert scores.size() == (1, candidates_dict["output_size"])


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates_dict",
    [
        {"number": 0, "candidates": None, "output_size": None},
        {"number": 1.5, "candidates": None, "output_size": None},
    ],
)
def test_prediction_optimized_sasrec_invalid_num_candidates_to_score(
    item_user_sequential_dataset, train_sasrec_loader, candidates_dict
):
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_sasrec_loader)
    trainer.save_checkpoint("test.ckpt")

    with pytest.raises(ValueError):
        OptimizedSasRec.from_checkpoint(
            checkpoint_path="test.ckpt",
            mode="one_query",
            num_candidates_to_score=candidates_dict["number"],
        )
