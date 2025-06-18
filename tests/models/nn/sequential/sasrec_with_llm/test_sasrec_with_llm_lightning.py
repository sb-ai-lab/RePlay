import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.optimizer_utils import FatLRSchedulerFactory, FatOptimizerFactory
    from replay.models.nn.sequential.sasrec import SasRecPredictionBatch, SasRecPredictionDataset
    from replay.models.nn.sequential.sasrec_with_llm import SasRecLLM

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
    item_user_sequential_dataset,
    train_sasrec_llm_loader,
    val_sasrec_loader,
    loss_type,
    loss_sample_count,
):
    trainer = L.Trainer(max_epochs=1)
    model = SasRecLLM(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
        max_seq_len=5,
        hidden_size=64,
        loss_type=loss_type,
        loss_sample_count=loss_sample_count,
        profile_distil_epochs=0,
    )
    trainer.fit(model, train_sasrec_llm_loader, val_sasrec_loader)


@pytest.mark.torch
def test_init_sasrec_with_invalid_loss_type(item_user_sequential_dataset):
    with pytest.raises(NotImplementedError) as exc:
        SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema,
                  profile_emb_dim=1024,
                  max_seq_len=5,
                  hidden_size=64,
                  loss_type="")

    assert str(exc.value) == "Not supported loss_type"


@pytest.mark.torch
def test_train_sasrec_with_invalid_loss_type(item_user_sequential_dataset, train_sasrec_llm_loader):
    with pytest.raises(ValueError):
        trainer = L.Trainer(max_epochs=1)
        model = SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema,
                          profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
                          max_seq_len=5,
                          hidden_size=64
                          )
        model._loss_type = ""
        trainer.fit(model, train_dataloaders=train_sasrec_llm_loader)


@pytest.mark.torch
def test_prediction_sasrec(item_user_sequential_dataset, train_sasrec_llm_loader):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema,
                      profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
                      max_seq_len=5,
                      hidden_size=64)
    trainer.fit(model, train_sasrec_llm_loader)
    predicted = trainer.predict(model, pred_sasrec_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates",
    [torch.LongTensor([1]), torch.LongTensor([1, 2, 3, 4]), torch.LongTensor([0, 1, 2, 3, 4, 5]), None],
)
def test_prediction_sasrec_with_candidates(item_user_sequential_dataset, train_sasrec_llm_loader, candidates):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred, batch_size=1)
    trainer = L.Trainer(max_epochs=1)
    model = SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema,
                      profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
                      max_seq_len=5,
                      hidden_size=64)
    trainer.fit(model, train_sasrec_llm_loader)

    # test online inference with candidates
    for batch in pred_sasrec_loader:
        predicted = model.predict(batch, candidates)
        assert model.candidates_to_score is None
        if candidates is not None:
            assert predicted.size() == (1, candidates.shape[0])
        else:
            assert predicted.size() == (1, item_user_sequential_dataset.schema["item_id"].cardinality)

    # test offline inference with candidates
    model.candidates_to_score = candidates
    predicted = trainer.predict(model, pred_sasrec_loader)
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
def test_predictions_sasrec_equal_with_permuted_candidates(item_user_sequential_dataset, train_sasrec_llm_loader):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_sasrec_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema,
                      profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
                      max_seq_len=5,
                      hidden_size=64)
    trainer.fit(model, train_sasrec_llm_loader)

    sorted_candidates = torch.LongTensor([0, 1, 2, 3])
    permuted_candidates = torch.LongTensor([3, 0, 2, 1])
    _, ordering = torch.sort(permuted_candidates)

    model.candidates_to_score = sorted_candidates
    predictions_sorted_candidates = trainer.predict(model, pred_sasrec_loader)

    model.candidates_to_score = permuted_candidates
    predictions_permuted_candidates = trainer.predict(model, pred_sasrec_loader)
    for i in range(len(predictions_permuted_candidates)):
        assert torch.equal(predictions_permuted_candidates[i][:, ordering], predictions_sorted_candidates[i])


@pytest.mark.torch
@pytest.mark.parametrize(
    "candidates",
    [torch.FloatTensor([1]), torch.LongTensor([1] * 100000)],
)
def test_prediction_optimized_sasrec_invalid_candidates_to_score(
    item_user_sequential_dataset, train_sasrec_llm_loader, candidates
):
    trainer = L.Trainer(max_epochs=1)
    model = SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema,
                      profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
                      max_seq_len=5,
                      hidden_size=64
                      )
    trainer.fit(model, train_sasrec_llm_loader)

    with pytest.raises(ValueError):
        model.candidates_to_score = candidates


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
    if optimizer_factory:
        model = SasRecLLM(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            profile_emb_dim=1024,
            max_seq_len=5,
            hidden_size=64,
            lr_scheduler_factory=lr_scheduler_factory,
            optimizer_factory=optimizer_factory,
        )
    else:
        model = SasRecLLM(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            profile_emb_dim=1024,
            max_seq_len=5,
            hidden_size=64,
            lr_scheduler_factory=lr_scheduler_factory,
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
    train_sasrec_llm_loader,
    val_sasrec_loader,
    negative_sampling_strategy,
    negatives_sharing,
):
    trainer = L.Trainer(max_epochs=1)
    model = SasRecLLM(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
        negative_sampling_strategy=negative_sampling_strategy,
        negatives_sharing=negatives_sharing,
    )
    trainer.fit(model, train_sasrec_llm_loader, val_sasrec_loader)


@pytest.mark.torch
def test_not_implemented_sampling_strategy(item_user_sequential_dataset, train_sasrec_llm_loader, val_sasrec_loader):
    trainer = L.Trainer(max_epochs=1)
    model = SasRecLLM(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        profile_emb_dim=train_sasrec_llm_loader.dataset.user_profile_embeddings.shape[0],
        max_seq_len=5,
        hidden_size=64,
        loss_sample_count=6,
    )
    model._negative_sampling_strategy = ""
    with pytest.raises(NotImplementedError):
        trainer.fit(model, train_sasrec_llm_loader, val_sasrec_loader)


@pytest.mark.torch
def test_model_predict_with_nn_parallel(item_user_sequential_dataset, simple_masks):
    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    model = SasRecLLM(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        profile_emb_dim=1024,
        max_seq_len=5,
        hidden_size=64,
        loss_sample_count=6,
    )

    model._model = torch.nn.DataParallel(model._model)
    model._model_predict({"item_id": item_sequences}, padding_mask)


@pytest.mark.torch
def test_sasrec_get_embeddings(tensor_schema):
    model = SasRecLLM(tensor_schema, profile_emb_dim=1024)
    model_ti = SasRecLLM(tensor_schema, profile_emb_dim=1024, ti_modification=True)

    model_embeddings = model.get_all_embeddings()
    model_ti_embeddings = model_ti.get_all_embeddings()

    model_item_embedding = model_embeddings["item_embedding"]
    model_ti_item_embedding = model_ti_embeddings["item_embedding"]

    # Check number of embeddings for each layer in sasrec
    assert len(model_embeddings) == 2
    assert len(model_ti_embeddings) == 5

    # Check types
    assert isinstance(model_item_embedding, torch.Tensor)
    assert isinstance(model_ti_item_embedding, torch.Tensor)

    # Ensure we got copies
    assert id(model_item_embedding) != id(model._model.item_embedder.item_emb)
    assert id(model_ti_item_embedding) != id(model_ti._model.item_embedder.item_emb)

    # Ensure we got same values
    assert torch.eq(model_item_embedding, model._model.item_embedder.item_emb.weight.data[:-1, :]).all()
    assert torch.eq(model_ti_item_embedding, model_ti._model.item_embedder.item_emb.weight.data[:-1, :]).all()


@pytest.mark.torch
def test_sasrec_fine_tuning_on_new_items_by_size(fitted_sasrec, new_items_dataset):
    model, tokenizer = fitted_sasrec
    old_items_data = model._model.item_embedder.item_emb.weight.data
    shape = old_items_data.shape
    old_vocab_size = len(tokenizer.item_id_encoder.mapping["item_id"])

    tokenizer.item_id_encoder.partial_fit(new_items_dataset)
    new_vocab_size = len(tokenizer.item_id_encoder.mapping["item_id"])

    model.set_item_embeddings_by_size(new_vocab_size)
    new_items_data = model._model.item_embedder.item_emb.weight.data
    new_shape = new_items_data.shape

    assert shape == (5, 50)
    assert old_vocab_size == 4
    assert new_vocab_size == 5
    assert new_shape == (6, 50)
    for item_num in range(old_vocab_size):
        assert torch.eq(old_items_data[item_num], new_items_data[item_num]).all()


@pytest.mark.torch
def test_sasrec_fine_tuning_on_new_items_by_tensor(fitted_sasrec, new_items_dataset):
    model, tokenizer = fitted_sasrec
    tokenizer.item_id_encoder.partial_fit(new_items_dataset)

    new_items_tensor = torch.rand(5, 50)
    model.set_item_embeddings_by_tensor(new_items_tensor)

    new_items_data = model._model.item_embedder.item_emb.weight.data
    new_shape = new_items_data.shape

    assert new_shape == (6, 50)
    for item_num in range(new_shape[0] - 1):
        assert torch.eq(new_items_tensor[item_num], new_items_data[item_num]).all()


@pytest.mark.torch
def test_sasrec_fine_tuning_on_new_items_by_appending(fitted_sasrec, new_items_dataset):
    model, tokenizer = fitted_sasrec
    tokenizer.item_id_encoder.partial_fit(new_items_dataset)
    old_items_tensor = model._model.item_embedder.item_emb.weight.data

    only_new_items_tensor = torch.rand(1, 50)
    model.append_item_embeddings(only_new_items_tensor)

    new_items_data = model._model.item_embedder.item_emb.weight.data
    new_shape = new_items_data.shape

    assert new_shape == (6, 50)
    for item_num in range(old_items_tensor.shape[0] - 1):
        assert torch.eq(old_items_tensor[item_num], new_items_data[item_num]).all()
    assert torch.eq(only_new_items_tensor, new_items_data[4]).all()


@pytest.mark.torch
def test_sasrec_fine_tuning_save_load(fitted_sasrec, new_items_dataset, train_sasrec_llm_loader):
    model, tokenizer = fitted_sasrec
    trainer = L.Trainer(max_epochs=1)
    tokenizer.item_id_encoder.partial_fit(new_items_dataset)
    new_vocab_size = len(tokenizer.item_id_encoder.mapping["item_id"])
    model.set_item_embeddings_by_size(new_vocab_size)
    trainer.fit(model, train_sasrec_llm_loader)
    trainer.save_checkpoint("test.ckpt")
    best_model = SasRecLLM.load_from_checkpoint("test.ckpt")

    assert best_model.get_all_embeddings()["item_embedding"].shape[0] == new_vocab_size


@pytest.mark.torch
def test_sasrec_fine_tuning_errors(fitted_sasrec):
    model, _ = fitted_sasrec

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
def test_sasrec_get_init_parameters(fitted_sasrec):
    model, _ = fitted_sasrec
    params = model.hparams

    assert params["tensor_schema"].item().cardinality == 4
    assert params["max_seq_len"] == 5
    assert params["hidden_size"] == 50


def test_predict_step_with_small_seq_len(item_user_num_sequential_dataset, simple_masks):
    item_sequences, padding_mask, _, _ = simple_masks

    model = SasRecLLM(
        tensor_schema=item_user_num_sequential_dataset._tensor_schema,
        profile_emb_dim=1024,
        max_seq_len=10,
        hidden_size=64,
        loss_sample_count=6,
    )

    batch = SasRecPredictionBatch(
        torch.arange(0, 4),
        padding_mask,
        {"item_id": item_sequences, "num_feature": item_sequences},
    )
    model.predict_step(batch, 0)


@pytest.mark.torch
def test_predict_step_with_big_seq_len(item_user_sequential_dataset, simple_masks):
    item_sequences, padding_mask, _, _ = simple_masks

    model = SasRecLLM(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        profile_emb_dim=1024,
        max_seq_len=3,
        hidden_size=64,
        loss_sample_count=6,
    )

    batch = SasRecPredictionBatch(torch.arange(0, 4), padding_mask, {"item_id": item_sequences})
    with pytest.raises(ValueError):
        model.predict_step(batch, 0)


@pytest.mark.torch
def test_sasrec_get_set_optim_factory(item_user_sequential_dataset):
    optim_factory = FatOptimizerFactory()
    model = SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema,
                      profile_emb_dim=1024,
                      optimizer_factory=optim_factory
                      )

    assert model.optimizer_factory is optim_factory
    new_factory = FatOptimizerFactory(learning_rate=0.1)
    model.optimizer_factory = new_factory
    assert model.optimizer_factory is new_factory


@pytest.mark.torch
def test_sasrec_set_invalid_optim_factory(item_user_sequential_dataset):
    model = SasRecLLM(tensor_schema=item_user_sequential_dataset._tensor_schema, profile_emb_dim=1024)
    new_factory = "Let's say it's an optimizer_factory"
    with pytest.raises(ValueError):
        model.optimizer_factory = new_factory
