import lightning as L
import pytest
import torch

from replay.nn.lightning import LightningModule
from replay.nn.lightning.optimizer import OptimizerFactory
from replay.nn.lightning.scheduler import LambdaLRSchedulerFactory


def test_training_bert4rec_with_different_losses(
    bert4rec_parametrized,
    parquet_module_with_bert4rec_negative_sampling_transform,
):
    bert4rec = LightningModule(
        bert4rec_parametrized,
        optimizer_factory=OptimizerFactory(),
        lr_scheduler_factory=LambdaLRSchedulerFactory(warmup_steps=1),
    )
    trainer = L.Trainer(max_epochs=1, accelerator="cpu")
    trainer.fit(bert4rec, datamodule=parquet_module_with_bert4rec_negative_sampling_transform)


def test_bert4rec_with_default_transform(
    bert4rec_model,
    parquet_module_with_default_bert4rec_transform,
) -> None:
    bert4rec = LightningModule(
        bert4rec_model,
        optimizer_factory=OptimizerFactory(),
    )
    trainer = L.Trainer(max_epochs=1, accelerator="cpu")
    trainer.fit(bert4rec, datamodule=parquet_module_with_default_bert4rec_transform)
    trainer.test(bert4rec, datamodule=parquet_module_with_default_bert4rec_transform)


def test_bert4rec_checkpointing(bert4rec_model, parquet_module_with_default_bert4rec_transform, tmp_path):
    bert4rec = LightningModule(bert4rec_model)
    trainer = L.Trainer(max_epochs=1, accelerator="cpu")
    trainer.fit(bert4rec, datamodule=parquet_module_with_default_bert4rec_transform)

    ckpt_path = tmp_path / "checkpoints/last.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded_bert4rec = LightningModule.load_from_checkpoint(ckpt_path, model=bert4rec_model)

    batch = parquet_module_with_default_bert4rec_transform.compiled_transforms["train"](
        next(iter(parquet_module_with_default_bert4rec_transform.train_dataloader()))
    )

    bert4rec.eval()
    loaded_bert4rec.eval()

    output1 = bert4rec(batch)
    output2 = loaded_bert4rec(batch)

    torch.testing.assert_close(output1["logits"], output2["logits"])
    torch.testing.assert_close(output1["hidden_states"][0], output2["hidden_states"][0])


@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.LongTensor([1]), torch.LongTensor([1, 2]), torch.arange(0, 15, dtype=torch.long), None],
)
def test_bert4rec_prediction_with_candidates(
    tensor_schema_with_equal_embedding_dims,
    bert4rec_model,
    parquet_module_with_default_bert4rec_transform,
    candidates_to_score,
):
    bert4rec = LightningModule(bert4rec_model)
    trainer = L.Trainer(inference_mode=True, accelerator="cpu")
    bert4rec.eval()
    bert4rec.candidates_to_score = candidates_to_score
    predictions = trainer.predict(bert4rec, datamodule=parquet_module_with_default_bert4rec_transform)

    if candidates_to_score is not None:
        assert torch.equal(bert4rec.candidates_to_score, candidates_to_score)
    else:
        assert bert4rec.candidates_to_score is None

    for pred in predictions[:-1]:
        if candidates_to_score is None:
            assert pred["logits"].size() == (
                parquet_module_with_default_bert4rec_transform.batch_size,
                tensor_schema_with_equal_embedding_dims["item_id"].cardinality,
            )
        else:
            assert pred["logits"].size() == (
                parquet_module_with_default_bert4rec_transform.batch_size,
                candidates_to_score.shape[0],
            )


@pytest.mark.parametrize("random_seed", [0, 1, 2, 3, 4])
def test_predictions_bert4rec_equal_with_permuted_candidates(
    tensor_schema_with_equal_embedding_dims,
    bert4rec_model,
    parquet_module_with_default_bert4rec_transform,
    random_seed,
):
    bert4rec = LightningModule(bert4rec_model)
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    items_cardinality = tensor_schema_with_equal_embedding_dims["item_id"].cardinality
    num_samples = torch.randint(low=1, high=items_cardinality, size=(1,), generator=generator)

    permuted_candidates = torch.multinomial(
        input=torch.ones(items_cardinality), num_samples=num_samples, replacement=False, generator=generator
    )
    sorted_candidates, ordering = torch.sort(permuted_candidates)

    trainer = L.Trainer(inference_mode=True, accelerator="cpu")
    bert4rec.eval()
    bert4rec.candidates_to_score = sorted_candidates
    predictions_sorted_candidates = trainer.predict(
        bert4rec, datamodule=parquet_module_with_default_bert4rec_transform
    )

    bert4rec.candidates_to_score = permuted_candidates
    predictions_permuted_candidates = trainer.predict(
        bert4rec, datamodule=parquet_module_with_default_bert4rec_transform
    )

    for i in range(len(predictions_permuted_candidates)):
        assert torch.allclose(
            predictions_permuted_candidates[i]["logits"][:, ordering],
            predictions_sorted_candidates[i]["logits"],
            rtol=1e-3,
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "candidates_to_score",
    [torch.FloatTensor([1]), torch.BoolTensor([1, 0]), torch.LongTensor([1, 1])],
    ids=["Float tensor", "Bool tensor", "Tensor with non-unique values"],
)
def test_bert4rec_prediction_invalid_candidates_to_score(
    bert4rec_model,
    parquet_module_with_default_bert4rec_transform,
    candidates_to_score,
):
    bert4rec = LightningModule(bert4rec_model)
    trainer = L.Trainer(max_epochs=1, accelerator="cpu")
    trainer.fit(bert4rec, datamodule=parquet_module_with_default_bert4rec_transform)

    trainer = L.Trainer(inference_mode=True, accelerator="cpu")

    with pytest.raises((RuntimeError, ValueError)):
        bert4rec.candidates_to_score = candidates_to_score
        trainer.predict(bert4rec, datamodule=parquet_module_with_default_bert4rec_transform)
