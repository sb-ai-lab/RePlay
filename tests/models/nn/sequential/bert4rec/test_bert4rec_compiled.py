import os

import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.bert4rec import (
        Bert4Rec,
        Bert4RecPredictionDataset,
    )
    from replay.models.nn.sequential.bert4rec.optimized_model import Bert4RecCompiled


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
def test_prediction_optimized_bert4rec(
    item_user_sequential_dataset, train_bert_loader, num_candidates, candidates, mode, batch_size, tmp_path
):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_bert4rec_loader = torch.utils.data.DataLoader(pred, batch_size=batch_size)
    cardinality = item_user_sequential_dataset.schema["item_id"].cardinality

    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_bert_loader)
    trainer.save_checkpoint(tmp_path / "test.ckpt")

    opt_model = Bert4RecCompiled.compile(
        model=(tmp_path / "test.ckpt"),
        mode=mode,
        batch_size=batch_size if mode != "dynamic_batch_size" else None,
        num_candidates_to_score=num_candidates,
    )

    score_size = candidates.shape[0] if candidates is not None else cardinality

    for batch in pred_bert4rec_loader:
        if candidates is not None:
            scores = opt_model.predict(batch=batch, candidates_to_score=candidates)
        else:
            scores = opt_model.predict(batch=batch)
        assert scores.shape == (batch.padding_mask.shape[0], score_size)


@pytest.mark.torch
def test_predictions_optimized_bert4rec_equal_with_permuted_candidates(item_user_sequential_dataset, train_bert_loader):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_bert4rec_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64)
    trainer.fit(model, train_bert_loader)

    sorted_candidates = torch.LongTensor([0, 1, 2, 3])
    permuted_candidates = torch.LongTensor([3, 0, 2, 1])
    _, ordering = torch.sort(permuted_candidates)

    opt_model = Bert4RecCompiled.compile(
        model=model,
        num_candidates_to_score=sorted_candidates.shape[0],
    )

    for batch in pred_bert4rec_loader:
        predictions_sorted_candidates = opt_model.predict(batch=batch, candidates_to_score=sorted_candidates)
        predictions_permuted_candidates = opt_model.predict(batch=batch, candidates_to_score=permuted_candidates)
        assert torch.equal(predictions_permuted_candidates[:, ordering], predictions_sorted_candidates)


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
def test_prediction_optimized_bert4rec_invalid_candidates_to_score(
    item_user_sequential_dataset, train_bert_loader, num_candidates, candidates, tmp_path
):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_bert4rec_loader = torch.utils.data.DataLoader(pred)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_bert_loader)
    trainer.save_checkpoint(tmp_path / "test.ckpt")

    batch = next(iter(pred_bert4rec_loader))
    with pytest.raises(ValueError) as e:
        model = Bert4RecCompiled.compile(
            model=(tmp_path / "test.ckpt"),
            mode="one_query",
            num_candidates_to_score=num_candidates,
        )
        model.predict(batch, candidates)

    if num_candidates == 1:
        assert "Expected candidates to be of type ``torch.Tensor`` with dtype" in str(e.value)
    elif num_candidates is None:
        assert "If ``num_candidates_to_score`` is None," in str(e.value)
    else:
        assert "Expected num_candidates_to_score to be of type ``int``" in str(e.value)


@pytest.mark.torch
@pytest.mark.parametrize(
    "mode, batch_size, model_batch_size",
    [
        ("batch", 2, 3),
        ("batch", 2, 1),
    ],
)
def test_prediction_optimized_bert4rec_invalid_batch_in_batch_mode(
    item_user_sequential_dataset, train_bert_loader, mode, batch_size, model_batch_size, tmp_path
):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_bert4rec_loader = torch.utils.data.DataLoader(pred, batch_size=batch_size)

    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_bert_loader)
    trainer.save_checkpoint(tmp_path / "test.ckpt")

    opt_model = Bert4RecCompiled.compile(
        model=(tmp_path / "test.ckpt"),
        mode=mode,
        batch_size=model_batch_size,
    )

    with pytest.raises(ValueError):
        for batch in pred_bert4rec_loader:
            opt_model.predict(batch=batch)


@pytest.mark.torch
def test_optimized_bert4rec_invalid_mode():
    with pytest.raises(ValueError):
        Bert4RecCompiled.compile(model="some_path", mode="invalid_mode")


@pytest.mark.torch
def test_optimized_bert4rec_compile_from_different_sources(item_user_sequential_dataset, train_bert_loader, tmp_path):
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_bert_loader)
    trainer.save_checkpoint(tmp_path / "test.ckpt")

    opt_model1 = Bert4RecCompiled.compile(model=str(tmp_path / "test.ckpt"))
    opt_model2 = Bert4RecCompiled.compile(model=(tmp_path / "test.ckpt"))
    opt_model3 = Bert4RecCompiled.compile(model=model)
    assert str(opt_model1._model) == str(opt_model2._model) == str(opt_model3._model)


@pytest.mark.torch
def test_optimized_bert4rec_onnx_path_param(item_user_sequential_dataset, tmp_path):
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    path_to_save = tmp_path / "model.onnx"
    Bert4RecCompiled.compile(model=model, onnx_path=path_to_save)
    assert os.path.exists(path_to_save)


@pytest.mark.torch
@pytest.mark.parametrize("threads", [1, 2])
def test_optimized_bert4rec_num_threads_param(item_user_sequential_dataset, threads):
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    opt_model = Bert4RecCompiled.compile(model=model, num_threads=threads)
    assert opt_model._model.get_property("INFERENCE_NUM_THREADS") == threads
