import lightning as L
import pandas as pd
import pytest
import torch

from replay.nn.lightning import LightningModule
from replay.nn.lightning.callback import (
    ComputeMetricsCallback,
    HiddenStatesCallback,
    PandasTopItemsCallback,
    PolarsTopItemsCallback,
    SparkTopItemsCallback,
    TorchTopItemsCallback,
)
from replay.nn.lightning.postprocessor import SeenItemsFilter
from replay.utils import PandasDataFrame, PolarsDataFrame, SparkDataFrame
from replay.utils.session_handler import get_spark_session


@pytest.mark.parametrize(
    "callback_class",
    [
        pytest.param(TorchTopItemsCallback, marks=pytest.mark.torch),
        pytest.param(PandasTopItemsCallback, marks=pytest.mark.torch),
        pytest.param(PolarsTopItemsCallback, marks=pytest.mark.torch),
        pytest.param(SparkTopItemsCallback, marks=[pytest.mark.torch, pytest.mark.spark]),
    ],
)
@pytest.mark.parametrize("is_postprocessor", [False, True])
@pytest.mark.parametrize(
    "candidates",
    [
        torch.LongTensor([0]),
        torch.LongTensor([2, 13]),
        torch.LongTensor([1, 2, 3, 4]),
        torch.LongTensor([11, 6, 9]),
        None,
    ],
)
def test_prediction_callbacks_fast_forward(
    parquet_module,
    tensor_schema,
    sasrec_model,
    callback_class,
    is_postprocessor,
    candidates,
):
    cardinality = tensor_schema["item_id"].cardinality

    kwargs = {
        "top_k": 1,
        "postprocessors": (
            [SeenItemsFilter(item_count=cardinality, seen_items_column="seen_ids")] if is_postprocessor else None
        ),
    }
    kwargs["query_column"] = "user_id"
    if callback_class != TorchTopItemsCallback:
        kwargs["item_column"] = "item_id"
    if callback_class == SparkTopItemsCallback:
        kwargs["spark_session"] = get_spark_session()
        kwargs["rating_column"] = "score"

    callback = callback_class(**kwargs)

    model = LightningModule(sasrec_model)
    if candidates is not None:
        model.candidates_to_score = candidates

    trainer = L.Trainer(inference_mode=True, callbacks=[callback])
    predicted = trainer.predict(model, datamodule=parquet_module)

    assert len(predicted) == len(parquet_module.predict_dataloader())

    score_size = cardinality if candidates is None else candidates.shape[0]
    assert predicted[0]["logits"].size() == (parquet_module.batch_size, score_size)

    if callback_class == TorchTopItemsCallback:
        users, items, scores = callback.get_result()
        assert isinstance(users, torch.LongTensor)
        assert isinstance(items, torch.LongTensor)
        assert isinstance(scores, torch.Tensor)
        if candidates is not None:
            assert set(items.flatten().numpy()) <= set(candidates.numpy())
    else:
        result = callback.get_result()
        if callback_class == PandasTopItemsCallback:
            result_type = PandasDataFrame
        elif callback_class == PolarsTopItemsCallback:
            result_type = PolarsDataFrame
        elif callback_class == SparkTopItemsCallback:
            result_type = SparkDataFrame
        assert isinstance(result, result_type)


@pytest.mark.torch
@pytest.mark.parametrize(
    "metrics, postprocessor",
    [
        (["coverage", "precision"], SeenItemsFilter),
        (["coverage"], SeenItemsFilter),
        (["coverage", "precision"], None),
        (["coverage"], None),
    ],
)
@pytest.mark.parametrize(
    "parquet_fixture",
    (
        "parquet_module",
        "parquet_module_with_multiple_val_paths",
    ),
)
def test_validation_callbacks(
    parquet_fixture,
    tensor_schema,
    sasrec_model,
    metrics,
    postprocessor,
    request: pytest.FixtureRequest,
):
    parquet_module = request.getfixturevalue(parquet_fixture)
    cardinality = tensor_schema["item_id"].cardinality

    callback = ComputeMetricsCallback(
        metrics=metrics,
        ks=[1],
        item_count=cardinality,
        postprocessors=(
            [postprocessor(item_count=cardinality, seen_items_column="seen_ids")] if postprocessor else None
        ),
    )
    model = LightningModule(sasrec_model)

    trainer = L.Trainer(max_epochs=1, callbacks=[callback])
    trainer.fit(model, datamodule=parquet_module)

    for metric in metrics:
        assert any(key.startswith(metric) for key in trainer.callback_metrics.keys())

    trainer = L.Trainer(callbacks=[callback], inference_mode=True)
    model.eval()
    trainer.test(model, datamodule=parquet_module)

    for metric in metrics:
        assert any(key.startswith(metric) for key in trainer.callback_metrics.keys())


def test_query_embeddings_callback(sasrec_model, parquet_module, parquet_module_path):
    model = LightningModule(sasrec_model)

    callback = HiddenStatesCallback(hidden_state_index=0)
    trainer = L.Trainer(inference_mode=True, callbacks=[callback])
    trainer.predict(model, datamodule=parquet_module, return_predictions=False)

    sequential_data = pd.read_parquet(parquet_module_path)
    embs = callback.get_result()

    assert embs.shape[0] == (sequential_data.shape[0])
