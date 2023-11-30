import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.optimizer_utils import FatLRSchedulerFactory, FatOptimizerFactory
    from replay.models.nn.sequential.sasrec import SasRec, SasRecPredictionDataset

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
def test_training_sasrec_with_different_losses(
    item_user_sequential_dataset, train_sasrec_loader, val_sasrec_loader, loss_type, loss_sample_count
):
    trainer = L.Trainer(max_epochs=1)
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type=loss_type,
        loss_sample_count=loss_sample_count,
    )
    trainer.fit(model, train_sasrec_loader, val_sasrec_loader)


@pytest.mark.torch
def test_init_sasrec_with_invalid_loss_type(item_user_sequential_dataset):
    with pytest.raises(NotImplementedError) as exc:
        SasRec(
            tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_type=""
        )

    assert str(exc.value) == "Not supported loss_type"


@pytest.mark.torch
def test_train_sasrec_with_invalid_loss_type(item_user_sequential_dataset, train_sasrec_loader):
    with pytest.raises(ValueError):
        trainer = L.Trainer(max_epochs=1)
        model = SasRec(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            max_seq_len=5,
            hidden_size=64,
        )
        model._loss_type = ""
        trainer.fit(model, train_dataloaders=train_sasrec_loader)


@pytest.mark.torch
def test_prediction_sasrec(item_user_sequential_dataset, train_sasrec_loader):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_sasrec_loader)
    predicted = trainer.predict(model, pred_sasrec_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)


@pytest.mark.torch
@pytest.mark.parametrize(
    "optimizer_factory, lr_scheduler_factory",
    [
        (None, None),
        (FatOptimizerFactory(), None),
        (None, FatLRSchedulerFactory()),
        (FatOptimizerFactory(), FatLRSchedulerFactory()),
    ],
)
def test_sasrec_configure_optimizers(item_user_sequential_dataset, optimizer_factory, lr_scheduler_factory):
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        lr_scheduler_factory=lr_scheduler_factory,
        optimizer_factory=optimizer_factory,
    )

    parameters = model.configure_optimizers()
    if isinstance(parameters, tuple):
        assert isinstance(parameters[0][0], torch.optim.Adam)
        assert isinstance(parameters[1][0], torch.optim.lr_scheduler.StepLR)
    else:
        assert isinstance(parameters, torch.optim.Adam)


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
    item_user_sequential_dataset,
    train_sasrec_loader,
    val_sasrec_loader,
    negative_sampling_strategy,
    negatives_sharing,
):
    trainer = L.Trainer(max_epochs=1)
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
        negative_sampling_strategy=negative_sampling_strategy,
        negatives_sharing=negatives_sharing,
    )
    trainer.fit(model, train_sasrec_loader, val_sasrec_loader)


@pytest.mark.torch
def test_not_implemented_sampling_strategy(item_user_sequential_dataset, train_sasrec_loader, val_sasrec_loader):
    trainer = L.Trainer(max_epochs=1)
    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_sample_count=6
    )
    model._negative_sampling_strategy = ""
    with pytest.raises(NotImplementedError):
        trainer.fit(model, train_sasrec_loader, val_sasrec_loader)


@pytest.mark.torch
def test_model_predict_with_nn_parallel(item_user_sequential_dataset, simple_masks):
    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    model = SasRec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_sample_count=6
    )

    model._model = torch.nn.DataParallel(model._model)
    model._model_predict({"item_id": item_sequences}, padding_mask)
