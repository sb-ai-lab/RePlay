import pytest

from replay.data import FeatureHint
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder
    from replay.models.nn.optimizer_utils import FatLRSchedulerFactory, FatOptimizerFactory
    from replay.models.nn.sequential.bert4rec import Bert4Rec, Bert4RecPredictionBatch, Bert4RecPredictionDataset

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "loss_type, loss_sample_count",
    [
        ("BCE", 6),
        ("CE", 6),
        ("CE_restricted", 6),
        ("BCE", None),
        ("CE", None),
        ("CE_restricted", None),
    ],
)
def test_training_bert4rec_with_different_losses(
    item_user_sequential_dataset, train_bert_loader, val_bert_loader, loss_type, loss_sample_count
):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type=loss_type,
        loss_sample_count=loss_sample_count,
    )
    trainer.fit(model, train_bert_loader, val_bert_loader)


@pytest.mark.torch
def test_deprecated_bert4rec_pipeline(
    item_user_sequential_dataset,
    deprecated_train_bert_loader,
    deprecated_val_bert_loader,
    deprecated_pred_bert_loader,
):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    with pytest.deprecated_call():
        trainer.fit(model, deprecated_train_bert_loader, deprecated_val_bert_loader)

    with pytest.deprecated_call():
        _ = trainer.predict(model, deprecated_pred_bert_loader)

    with pytest.deprecated_call():
        for batch in deprecated_pred_bert_loader:
            _ = model.predict(
                batch,
            )


@pytest.mark.torch
def test_init_bert4rec_with_invalid_loss_type(item_user_sequential_dataset):
    with pytest.raises(NotImplementedError) as exc:
        Bert4Rec(tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_type="")

    assert str(exc.value) == "Not supported loss_type"


@pytest.mark.torch
def test_train_bert4rec_with_invalid_loss_type(item_user_sequential_dataset, train_bert_loader):
    with pytest.raises(ValueError):
        trainer = L.Trainer(max_epochs=1)
        model = Bert4Rec(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            max_seq_len=5,
            hidden_size=64,
        )
        model._loss_type = ""
        trainer.fit(model, train_dataloaders=train_bert_loader)


@pytest.mark.torch
def test_prediction_bert4rec(item_user_sequential_dataset, train_bert_loader):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_bert_loader)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates",
    [torch.LongTensor([1]), torch.LongTensor([1, 2, 3, 4]), torch.LongTensor([0, 1, 2, 3, 4, 5])],
)
@pytest.mark.parametrize(
    "tying_head",
    [True, False],
)
def test_prediction_bert_with_candidates(item_user_sequential_dataset, train_bert_loader, candidates, tying_head):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_bert_loader = torch.utils.data.DataLoader(pred, batch_size=1)
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        enable_embedding_tying=tying_head,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_bert_loader)

    # test online inference with candidates
    for batch in pred_bert_loader:
        predicted = model.predict(batch, candidates)
        assert model.candidates_to_score is None
        if candidates is not None:
            assert predicted.size() == (1, candidates.shape[0])
        else:
            assert predicted.size() == (1, item_user_sequential_dataset.schema["item_id"].cardinality)

    # test offline inference with candidates
    model.candidates_to_score = candidates
    predicted = trainer.predict(model, pred_bert_loader)
    if candidates is not None:
        assert torch.equal(model.candidates_to_score, candidates)
    else:
        assert model.candidates_to_score is None

    for pred in predicted:
        if candidates is not None:
            assert pred.size() == (1, candidates.shape[0])
        else:
            assert pred.size() == (1, item_user_sequential_dataset.schema["item_id"].cardinality)


@pytest.mark.torch
def test_predictions_bert_equal_with_permuted_candidates(item_user_sequential_dataset, train_bert_loader):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_bert_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64)
    trainer.fit(model, train_bert_loader)

    sorted_candidates = torch.LongTensor([0, 1, 2, 3])
    permuted_candidates = torch.LongTensor([3, 0, 2, 1])
    _, ordering = torch.sort(permuted_candidates)

    model.candidates_to_score = sorted_candidates
    predictions_sorted_candidates = trainer.predict(model, pred_bert_loader)

    model.candidates_to_score = permuted_candidates
    predictions_permuted_candidates = trainer.predict(model, pred_bert_loader)
    for i in range(len(predictions_permuted_candidates)):
        assert torch.equal(predictions_permuted_candidates[i][:, ordering], predictions_sorted_candidates[i])


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates",
    [torch.FloatTensor([1]), torch.LongTensor([1] * 100000)],
)
def test_prediction_optimized_bert_invalid_candidates_to_score(
    item_user_sequential_dataset, train_bert_loader, candidates
):
    model = Bert4Rec(tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64)
    with pytest.raises(ValueError):
        model.candidates_to_score = candidates


@pytest.mark.torch
@pytest.mark.parametrize(
    "optimizer_factory, lr_scheduler_factory, optimizer_type, lr_scheduler_type",
    [
        (None, None, torch.optim.Adam, None),
        (FatOptimizerFactory(), None, torch.optim.Adam, None),
        (None, FatLRSchedulerFactory(), torch.optim.Adam, torch.optim.lr_scheduler.StepLR),
        (FatOptimizerFactory("sgd"), None, torch.optim.SGD, None),
        (FatOptimizerFactory(), FatLRSchedulerFactory(), torch.optim.Adam, torch.optim.lr_scheduler.StepLR),
    ],
)
def test_bert4rec_configure_optimizers(
    item_user_sequential_dataset,
    optimizer_factory,
    lr_scheduler_factory,
    optimizer_type,
    lr_scheduler_type,
):
    if optimizer_factory:
        model = Bert4Rec(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            max_seq_len=5,
            hidden_size=64,
            lr_scheduler_factory=lr_scheduler_factory,
            optimizer_factory=optimizer_factory,
        )
    else:
        model = Bert4Rec(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            max_seq_len=5,
            hidden_size=64,
            lr_scheduler_factory=lr_scheduler_factory,
        )

    parameters = model.configure_optimizers()
    if isinstance(parameters, tuple):
        assert isinstance(parameters[0][0], optimizer_type)
        assert isinstance(parameters[1][0], lr_scheduler_type)
    else:
        assert isinstance(parameters, optimizer_type)


@pytest.mark.torch
def test_bert4rec_configure_wrong_optimizer(item_user_sequential_dataset):
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        optimizer_factory=FatOptimizerFactory(""),
    )

    with pytest.raises(ValueError) as exc:
        model.configure_optimizers()

    assert str(exc.value) == "Unexpected optimizer"


@pytest.mark.torch
@pytest.mark.parametrize(
    "negative_sampling_strategy, negatives_sharing",
    [
        ("global_uniform", False),
        ("global_uniform", True),
        ("inbatch", False),
        ("inbatch", True),
    ],
)
def test_different_sampling_strategies(
    item_user_sequential_dataset, train_bert_loader, val_bert_loader, negative_sampling_strategy, negatives_sharing
):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
        negative_sampling_strategy=negative_sampling_strategy,
        negatives_sharing=negatives_sharing,
    )
    trainer.fit(model, train_bert_loader, val_bert_loader)


@pytest.mark.torch
def test_not_implemented_sampling_strategy(item_user_sequential_dataset, train_bert_loader, val_bert_loader):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_sample_count=6
    )
    model._negative_sampling_strategy = ""
    with pytest.raises(NotImplementedError):
        trainer.fit(model, train_bert_loader, val_bert_loader)


@pytest.mark.torch
def test_model_predict_with_nn_parallel(item_user_sequential_dataset, simple_masks):
    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_sample_count=6
    )

    model._model = torch.nn.DataParallel(model._model)
    model._model_predict({"item_id": item_sequences}, padding_mask, tokens_mask)


@pytest.mark.torch
def test_bert4rec_get_embeddings():
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            "some_feature",
            cardinality=6,
            is_seq=True,
        )
        .build()
    )
    model = Bert4Rec(schema, max_seq_len=5, enable_embedding_tying=True)
    model_embeddings = model.get_all_embeddings()
    model_item_embedding = model_embeddings["item_embedding"]

    assert len(model_embeddings) == 3
    assert isinstance(model_item_embedding, torch.Tensor)
    assert id(model_item_embedding) != id(model._model.item_embedder)
    assert torch.eq(model_item_embedding, model._model.item_embedder.item_embeddings.data).all()


@pytest.mark.torch
@pytest.mark.parametrize("fitted_bert4rec_model", [("fitted_bert4rec"), ("fitted_bert4rec_enable_embedding_tying")])
def test_bert4rec_fine_tuning_on_new_items_by_size(request, fitted_bert4rec_model, new_items_dataset):
    fitted_bert4rec = request.getfixturevalue(fitted_bert4rec_model)

    model, tokenizer = fitted_bert4rec
    old_items_data = model._model.item_embedder.item_embeddings.data
    shape = old_items_data.shape
    old_vocab_size = len(tokenizer.item_id_encoder.mapping["item_id"])

    tokenizer.item_id_encoder.partial_fit(new_items_dataset)
    new_vocab_size = len(tokenizer.item_id_encoder.mapping["item_id"])

    model.set_item_embeddings_by_size(new_vocab_size)
    new_items_data = model._model.item_embedder.item_embeddings.data
    new_shape = new_items_data.shape

    assert shape == (4, 64)
    assert new_shape == (5, 64)
    assert old_vocab_size == 4
    assert new_vocab_size == 5
    for item_num in range(old_vocab_size):
        assert torch.eq(old_items_data[item_num], new_items_data[item_num]).all()


@pytest.mark.torch
@pytest.mark.parametrize("fitted_bert4rec_model", [("fitted_bert4rec"), ("fitted_bert4rec_enable_embedding_tying")])
def test_bert4rec_fine_tuning_on_new_items_by_tensor(request, fitted_bert4rec_model, new_items_dataset):
    fitted_bert4rec = request.getfixturevalue(fitted_bert4rec_model)

    model, tokenizer = fitted_bert4rec
    old_vocab_size = len(tokenizer.item_id_encoder.mapping["item_id"])

    new_items_tensor = torch.rand(5, 64)
    tokenizer.item_id_encoder.partial_fit(new_items_dataset)

    model.set_item_embeddings_by_tensor(new_items_tensor)
    new_items_data = model._model.item_embedder.item_embeddings.data
    new_shape = new_items_data.shape

    assert new_shape == (5, 64)
    for item_num in range(old_vocab_size):
        assert torch.eq(new_items_tensor[item_num], new_items_data[item_num]).all()


@pytest.mark.torch
@pytest.mark.parametrize("fitted_bert4rec_model", [("fitted_bert4rec"), ("fitted_bert4rec_enable_embedding_tying")])
def test_bert4rec_fine_tuning_on_new_items_by_appending(request, fitted_bert4rec_model, new_items_dataset):
    fitted_bert4rec = request.getfixturevalue(fitted_bert4rec_model)

    model, tokenizer = fitted_bert4rec
    old_items_tensor = model._model.item_embedder.item_embeddings.data

    only_new_items_tensor = torch.rand(1, 64)
    tokenizer.item_id_encoder.partial_fit(new_items_dataset)

    model.append_item_embeddings(only_new_items_tensor)
    new_items_data = model._model.item_embedder.item_embeddings.data
    new_shape = new_items_data.shape

    assert new_shape == (5, 64)
    for item_num in range(old_items_tensor.shape[0] - 1):
        assert torch.eq(old_items_tensor[item_num], new_items_data[item_num]).all()
    assert torch.eq(only_new_items_tensor, new_items_data[4]).all()


@pytest.mark.torch
def test_bert4rec_fine_tuning_save_load(fitted_bert4rec, new_items_dataset, train_bert_loader):
    model, tokenizer = fitted_bert4rec
    trainer = L.Trainer(max_epochs=1)
    tokenizer.item_id_encoder.partial_fit(new_items_dataset)
    new_vocab_size = len(tokenizer.item_id_encoder.mapping["item_id"])
    model.set_item_embeddings_by_size(new_vocab_size)
    trainer.fit(model, train_bert_loader)
    trainer.save_checkpoint("bert_test.ckpt")
    best_model = Bert4Rec.load_from_checkpoint("bert_test.ckpt")

    assert best_model.get_all_embeddings()["item_embedding"].shape[0] == new_vocab_size


@pytest.mark.torch
def test_bert4rec_fine_tuning_errors(fitted_bert4rec):
    model, _ = fitted_bert4rec

    with pytest.raises(ValueError):
        model.set_item_embeddings_by_size(3)
    with pytest.raises(ValueError):
        model.set_item_embeddings_by_tensor(torch.rand(1, 1, 1))
    with pytest.raises(ValueError):
        model.set_item_embeddings_by_tensor(torch.rand(3, 50))
    with pytest.raises(ValueError):
        model.set_item_embeddings_by_tensor(torch.rand(4, 1))
    with pytest.raises(ValueError):
        model.append_item_embeddings(torch.rand(1, 1, 1))
    with pytest.raises(ValueError):
        model.append_item_embeddings(torch.rand(1, 1))


@pytest.mark.torch
def test_predict_step_with_small_seq_len(item_user_num_sequential_dataset, simple_masks):
    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    model = Bert4Rec(
        tensor_schema=item_user_num_sequential_dataset._tensor_schema,
        max_seq_len=10,
        hidden_size=64,
        loss_sample_count=6,
    )

    batch = Bert4RecPredictionBatch(
        torch.arange(0, 4), padding_mask, {"item_id": item_sequences, "num_feature": item_sequences}, tokens_mask
    )
    model.predict_step(batch, 0)


@pytest.mark.torch
def test_predict_step_with_big_seq_len(item_user_sequential_dataset, simple_masks):
    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=3, hidden_size=64, loss_sample_count=6
    )

    batch = Bert4RecPredictionBatch(torch.arange(0, 4), padding_mask, {"item_id": item_sequences}, tokens_mask)
    with pytest.raises(ValueError):
        model.predict_step(batch, 0)


@pytest.mark.torch
def test_bert4rec_get_set_optim_factory(item_user_sequential_dataset):
    optim_factory = FatOptimizerFactory()
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        optimizer_factory=optim_factory,
    )

    assert model.optimizer_factory is optim_factory
    new_factory = FatOptimizerFactory(learning_rate=0.1)
    model.optimizer_factory = new_factory
    assert model.optimizer_factory is new_factory


@pytest.mark.torch
def test_bert4rec_set_invalid_optim_factory(item_user_sequential_dataset):
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
    )
    new_factory = "Let's say it's an optimizer_factory"
    with pytest.raises(ValueError):
        model.optimizer_factory = new_factory
