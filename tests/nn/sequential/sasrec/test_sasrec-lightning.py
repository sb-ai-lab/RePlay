import lightning as L
import pytest
import torch

from replay.nn.lightning import LightningModule
from replay.nn.lightning.optimizer import OptimizerFactory
from replay.nn.lightning.scheduler import LambdaLRSchedulerFactory


def test_training_sasrec_with_different_losses(sasrec_parametrized, parquet_module):
    sasrec = LightningModule(
        sasrec_parametrized,
        optimizer_factory=OptimizerFactory(),
        lr_scheduler_factory=LambdaLRSchedulerFactory(warmup_steps=1),
    )
    trainer = L.Trainer(max_epochs=2)
    trainer.fit(sasrec, datamodule=parquet_module)


def test_sasrec_with_default_transform(sasrec_model_only_items, parquet_module_with_default_sasrec_transform):
    sasrec = LightningModule(
        sasrec_model_only_items,
        optimizer_factory=OptimizerFactory(),
    )
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module_with_default_sasrec_transform)
    trainer.test(sasrec, datamodule=parquet_module_with_default_sasrec_transform)


def test_sasrec_checkpointing(sasrec_model, parquet_module, tmp_path):
    sasrec = LightningModule(sasrec_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    ckpt_path = tmp_path / "checkpoints/last.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded_sasrec = LightningModule.load_from_checkpoint(ckpt_path, model=sasrec_model)

    batch = parquet_module.compiled_transforms["train"](next(iter(parquet_module.train_dataloader())))

    sasrec.eval()
    loaded_sasrec.eval()

    output1 = sasrec(batch)
    output2 = loaded_sasrec(batch)

    torch.testing.assert_close(output1["logits"], output2["logits"])
    torch.testing.assert_close(output1["hidden_states"][0], output2["hidden_states"][0])


@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.LongTensor([1]), torch.LongTensor([1, 2]), torch.arange(0, 40, dtype=torch.long), None],
)
def test_sasrec_prediction_with_candidates(tensor_schema, sasrec_model, parquet_module, candidates_to_score):
    sasrec = LightningModule(sasrec_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    sasrec.candidates_to_score = candidates_to_score
    trainer = L.Trainer(inference_mode=True)
    predictions = trainer.predict(sasrec, datamodule=parquet_module)

    if candidates_to_score is not None:
        assert torch.equal(sasrec.candidates_to_score, candidates_to_score)
    else:
        assert sasrec.candidates_to_score is None

    for pred in predictions[:-1]:
        if candidates_to_score is None:
            assert pred["logits"].size() == (parquet_module.batch_size, tensor_schema["item_id"].cardinality)
        else:
            assert pred["logits"].size() == (parquet_module.batch_size, candidates_to_score.shape[0])


@pytest.mark.parametrize("random_seed", [0, 1, 2, 3, 4])
def test_predictions_sasrec_equal_with_permuted_candidates(tensor_schema, sasrec_model, parquet_module, random_seed):
    sasrec = LightningModule(sasrec_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    items_cardinality = tensor_schema["item_id"].cardinality
    num_samples = torch.randint(low=1, high=items_cardinality, size=(1,), generator=generator)

    permuted_candidates = torch.multinomial(
        input=torch.ones(items_cardinality), num_samples=num_samples, replacement=False, generator=generator
    )
    sorted_candidates, ordering = torch.sort(permuted_candidates)

    trainer = L.Trainer(inference_mode=True)

    sasrec.candidates_to_score = sorted_candidates
    predictions_sorted_candidates = trainer.predict(sasrec, datamodule=parquet_module)

    sasrec.candidates_to_score = permuted_candidates
    predictions_permuted_candidates = trainer.predict(sasrec, datamodule=parquet_module)

    for i in range(len(predictions_permuted_candidates)):
        assert torch.equal(
            predictions_permuted_candidates[i]["logits"][:, ordering], predictions_sorted_candidates[i]["logits"]
        )


@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.FloatTensor([1]), torch.BoolTensor([1, 0]), torch.LongTensor([1, 1])],
    ids=["Float tensor", "Bool tensor", "Tensor with non-unique values"],
)
def test_sasrec_prediction_invalid_candidates_to_score(sasrec_model, parquet_module, candidates_to_score):
    sasrec = LightningModule(sasrec_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(sasrec, datamodule=parquet_module)

    trainer = L.Trainer(inference_mode=True)

    with pytest.raises((RuntimeError, ValueError)):
        sasrec.candidates_to_score = candidates_to_score
        trainer.predict(sasrec, datamodule=parquet_module)
