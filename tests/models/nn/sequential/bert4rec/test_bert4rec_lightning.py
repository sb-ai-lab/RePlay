import pytest

from replay.data import FeatureHint
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.optimizer_utils import FatLRSchedulerFactory, FatOptimizerFactory
    from replay.models.nn.sequential.bert4rec import Bert4Rec, Bert4RecPredictionDataset, Bert4RecPredictionBatch
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "loss_type, loss_sample_count",
    [
        ("BCE", 6),
        ("CE", 6),
        ("BCE", None),
        ("CE", None),
    ],
)
def test_training_bert4rec_with_different_losses(
    item_user_sequential_dataset, train_loader, val_loader, loss_type, loss_sample_count
):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type=loss_type,
        loss_sample_count=loss_sample_count,
    )
    trainer.fit(model, train_loader, val_loader)


@pytest.mark.torch
def test_init_bert4rec_with_invalid_loss_type(item_user_sequential_dataset):
    with pytest.raises(NotImplementedError) as exc:
        Bert4Rec(
            tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_type=""
        )

    assert str(exc.value) == "Not supported loss_type"


@pytest.mark.torch
def test_train_bert4rec_with_invalid_loss_type(item_user_sequential_dataset, train_loader):
    with pytest.raises(ValueError):
        trainer = L.Trainer(max_epochs=1)
        model = Bert4Rec(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            max_seq_len=5,
            hidden_size=64,
        )
        model._loss_type = ""
        trainer.fit(model, train_dataloaders=train_loader)


@pytest.mark.torch
def test_prediction_bert4rec(item_user_sequential_dataset, train_loader):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_loader)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)


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
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        lr_scheduler_factory=lr_scheduler_factory,
        optimizer_factory=optimizer_factory,
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
    item_user_sequential_dataset, train_loader, val_loader, negative_sampling_strategy, negatives_sharing
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
    trainer.fit(model, train_loader, val_loader)


@pytest.mark.torch
def test_not_implemented_sampling_strategy(item_user_sequential_dataset, train_loader, val_loader):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_sample_count=6
    )
    model._negative_sampling_strategy = ""
    with pytest.raises(NotImplementedError):
        trainer.fit(model, train_loader, val_loader)


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
    assert torch.eq(
        model_item_embedding,
        model._model.item_embedder.item_embeddings.data
    ).all()


@pytest.mark.torch
@pytest.mark.parametrize(
    "fitted_bert4rec_model",
    [
        ("fitted_bert4rec"),
        ("fitted_bert4rec_enable_embedding_tying")
    ]
)
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
@pytest.mark.parametrize(
    "fitted_bert4rec_model",
    [
        ("fitted_bert4rec"),
        ("fitted_bert4rec_enable_embedding_tying")
    ]
)
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
@pytest.mark.parametrize(
    "fitted_bert4rec_model",
    [
        ("fitted_bert4rec"),
        ("fitted_bert4rec_enable_embedding_tying")
    ]
)
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


def test_predict_step_with_small_seq_len(item_user_num_sequential_dataset, simple_masks):
    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    model = Bert4Rec(
        tensor_schema=item_user_num_sequential_dataset._tensor_schema, max_seq_len=10, hidden_size=64, loss_sample_count=6
    )

    batch = Bert4RecPredictionBatch(torch.arange(0, 4), padding_mask, {"item_id": item_sequences, "num_feature": item_sequences}, tokens_mask)
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
