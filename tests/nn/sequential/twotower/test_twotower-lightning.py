import lightning as L
import pytest
import torch

from replay.nn.lightning import LightningModule
from replay.nn.lightning.optimizer import OptimizerFactory
from replay.nn.lightning.scheduler import LambdaLRSchedulerFactory


def test_training_twotower_with_different_losses(twotower_parametrized, parquet_module):
    twotower = LightningModule(
        twotower_parametrized,
        optimizer_factory=OptimizerFactory(),
        lr_scheduler_factory=LambdaLRSchedulerFactory(warmup_steps=1),
    )
    trainer = L.Trainer(max_epochs=2)
    trainer.fit(twotower, datamodule=parquet_module)


def test_twotower_with_default_transform(twotower_model_only_items, parquet_module_with_default_twotower_transform):
    twotower = LightningModule(twotower_model_only_items)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, datamodule=parquet_module_with_default_twotower_transform)
    trainer.test(twotower, datamodule=parquet_module_with_default_twotower_transform)


def test_twotower_checkpointing(twotower_model, parquet_module, tmp_path):
    twotower = LightningModule(twotower_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, datamodule=parquet_module)

    ckpt_path = tmp_path / "checkpoints/last.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded_twotower = LightningModule.load_from_checkpoint(ckpt_path, model=twotower_model)

    batch = parquet_module.compiled_transforms["train"](next(iter(parquet_module.train_dataloader())))

    twotower.eval()
    loaded_twotower.eval()

    output1 = twotower(batch)
    output2 = loaded_twotower(batch)

    torch.testing.assert_close(output1["logits"], output2["logits"])
    torch.testing.assert_close(output1["hidden_states"][0], output2["hidden_states"][0])


@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.LongTensor([1]), torch.LongTensor([1, 2]), torch.arange(0, 40, dtype=torch.long), None],
)
def test_twotower_prediction_with_candidates(tensor_schema, twotower_model, parquet_module, candidates_to_score):
    twotower = LightningModule(twotower_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, datamodule=parquet_module)

    twotower.candidates_to_score = candidates_to_score
    trainer = L.Trainer(inference_mode=True)
    predictions = trainer.predict(twotower, datamodule=parquet_module)

    if candidates_to_score is not None:
        assert torch.equal(twotower.candidates_to_score, candidates_to_score)
    else:
        assert twotower.candidates_to_score is None

    for pred in predictions[:-1]:
        if candidates_to_score is None:
            assert pred["logits"].size() == (parquet_module.batch_size, tensor_schema["item_id"].cardinality - 1)
        else:
            assert pred["logits"].size() == (parquet_module.batch_size, candidates_to_score.shape[0])


@pytest.mark.parametrize("random_seed", [0, 1, 2, 3, 4])
def test_predictions_twotower_equal_with_permuted_candidates(
    tensor_schema, twotower_model, parquet_module, random_seed
):
    twotower = LightningModule(twotower_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, datamodule=parquet_module)

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    items_cardinality = tensor_schema["item_id"].cardinality
    num_samples = torch.randint(low=1, high=items_cardinality, size=(1,), generator=generator)

    permuted_candidates = torch.multinomial(
        input=torch.ones(items_cardinality), num_samples=num_samples, replacement=False, generator=generator
    )
    sorted_candidates, ordering = torch.sort(permuted_candidates)

    trainer = L.Trainer(inference_mode=True)

    twotower.candidates_to_score = sorted_candidates
    predictions_sorted_candidates = trainer.predict(twotower, datamodule=parquet_module)

    twotower.candidates_to_score = permuted_candidates
    predictions_permuted_candidates = trainer.predict(twotower, datamodule=parquet_module)

    for i in range(len(predictions_permuted_candidates)):
        assert torch.equal(
            predictions_permuted_candidates[i]["logits"][:, ordering], predictions_sorted_candidates[i]["logits"]
        )


@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.FloatTensor([1]), torch.BoolTensor([1, 0]), torch.LongTensor([1, 1])],
    ids=["Float tensor", "Bool tensor", "Tensor with non-unique values"],
)
def test_twotower_prediction_invalid_candidates_to_score(twotower_model, parquet_module, candidates_to_score):
    twotower = LightningModule(twotower_model)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(twotower, datamodule=parquet_module)

    trainer = L.Trainer(inference_mode=True)

    with pytest.raises((RuntimeError, ValueError, IndexError)):
        twotower.candidates_to_score = candidates_to_score
        trainer.predict(twotower, datamodule=parquet_module)
