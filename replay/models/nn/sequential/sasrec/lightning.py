import math
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

import lightning
import torch

from replay.data.nn import TensorMap, TensorSchema
from replay.models.nn.loss import ScalableCrossEntropyLoss, SCEParams
from replay.models.nn.optimizer_utils import FatOptimizerFactory, LRSchedulerFactory, OptimizerFactory

from .dataset import SasRecPredictionBatch, SasRecTrainingBatch, SasRecValidationBatch
from .model import SasRecModel


class SasRec(lightning.LightningModule):
    """
    SASRec Lightning module.

    You can get initialization parameters with attribute `hparams`
    for object of SasRec instance.
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        block_count: int = 2,
        head_count: int = 1,
        hidden_size: int = 50,
        max_seq_len: int = 200,
        dropout_rate: float = 0.2,
        ti_modification: bool = False,
        time_span: int = 256,
        loss_type: Literal["BCE", "CE", "SCE"] = "CE",
        loss_sample_count: Optional[int] = None,
        negative_sampling_strategy: str = "global_uniform",
        negatives_sharing: bool = False,
        optimizer_factory: OptimizerFactory = FatOptimizerFactory(),
        lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
        sce_params: Optional[SCEParams] = None,
    ):
        """
        :param tensor_schema: Tensor schema of features.
        :param block_count: Number of Transformer blocks.
            Default: ``2``.
        :param head_count: Number of Attention heads.
            Default: ``1``.
        :param hidden_size: Hidden size of transformer.
            Default: ``50``.
        :param max_seq_len: Max length of sequence.
            Default: ``200``.
        :param dropout_rate: Dropout rate.
            Default: ``0.2``.
        :param ti_modification: Enable time relation.
            Default: ``False``.
        :param time_span: Time span value.
            Default: ``256``.
        :param loss_type: Loss type. Possible values: ``"CE"``, ``"BCE"``, ``"SCE"``.
            Default: ``CE``.
        :param loss_sample_count (Optional[int]): Sample count to calculate loss.
            Suitable for ``"CE"`` and ``"BCE"`` loss functions.
            Default: ``None``.
        :param negative_sampling_strategy: Negative sampling strategy to calculate loss on sampled negatives.
            Is used when large count of items in dataset.
            Possible values: ``"global_uniform"``, ``"inbatch"``
            Default: ``global_uniform``.
        :param negatives_sharing: Apply negative sharing in calculating sampled logits.
            Default: ``False``.
        :param optimizer_factory: Optimizer factory.
            Default: ``FatOptimizerFactory``.
        :param lr_scheduler_factory: Learning rate schedule factory.
            Default: ``None``.
        :param sce_params: Dataclass with SCE parameters. Need to be defined if ``loss_type`` is ``SCE``.
            Default: ``None``.
        """
        super().__init__()
        self.save_hyperparameters()
        self._model = SasRecModel(
            schema=tensor_schema,
            num_blocks=block_count,
            num_heads=head_count,
            hidden_size=hidden_size,
            max_len=max_seq_len,
            dropout=dropout_rate,
            ti_modification=ti_modification,
            time_span=time_span,
        )
        self._loss_type = loss_type
        self._loss_sample_count = loss_sample_count
        self._negative_sampling_strategy = negative_sampling_strategy
        self._negatives_sharing = negatives_sharing
        self._optimizer_factory = optimizer_factory
        self._lr_scheduler_factory = lr_scheduler_factory
        self._sce_params = sce_params
        self._loss = self._create_loss()
        self._schema = tensor_schema
        assert negative_sampling_strategy in {"global_uniform", "inbatch"}
        if self._loss_type == "SCE":
            assert sce_params is not None, "You should define ``sce_params`` when using SCE loss function."

        item_count = tensor_schema.item_id_features.item().cardinality
        assert item_count
        self._vocab_size = item_count
        self.candidates_to_score = None

    def training_step(self, batch: SasRecTrainingBatch, batch_idx: int) -> torch.Tensor:
        """
        :param batch (SasRecTrainingBatch): Batch of training data.
        :param batch_idx (int): Batch index.

        :returns: Computed loss for batch.
        """
        if batch_idx % 100 == 0 and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(
        self,
        batch: SasRecPredictionBatch,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        :param batch: Batch of prediction data.
        :param batch_idx: Batch index.
        :param dataloader_idx: Dataloader index.

        :returns: Calculated scores.
        """
        batch = _prepare_prediction_batch(self._schema, self._model.max_len, batch)
        return self._model_predict(batch.features, batch.padding_mask)

    def predict(
        self,
        batch: SasRecPredictionBatch,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        :param batch: Batch of prediction data.
        :param candidates_to_score: Item ids to calculate scores.
            Default: ``None``.

        :returns: Calculated scores.
        """
        batch = _prepare_prediction_batch(self._schema, self._model.max_len, batch)
        return self._model_predict(batch.features, batch.padding_mask, candidates_to_score)

    def forward(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # pragma: no cover
        """
        :param feature_tensors: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param candidates_to_score: Item ids to calculate scores.
            Default: ``None``.

        :returns: Calculated scores.
        """
        return self._model_predict(feature_tensors, padding_mask, candidates_to_score)

    def validation_step(
        self,
        batch: SasRecValidationBatch,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        :param batch (SasRecValidationBatch): Batch of prediction data.
        :param batch_idx (int): Batch index.

        :returns: Calculated scores.
        """
        return self._model_predict(batch.features, batch.padding_mask)

    def configure_optimizers(self) -> Any:
        """
        :returns: Configured optimizer and lr scheduler.
        """
        optimizer = self._optimizer_factory.create(self._model.parameters())

        if self._lr_scheduler_factory is None:
            return optimizer

        lr_scheduler = self._lr_scheduler_factory.create(optimizer)
        return [optimizer], [lr_scheduler]

    def _model_predict(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: torch.LongTensor = None,
    ) -> torch.Tensor:
        model: SasRecModel
        model = cast(SasRecModel, self._model.module) if isinstance(self._model, torch.nn.DataParallel) else self._model
        candidates_to_score = self.candidates_to_score if candidates_to_score is None else candidates_to_score
        scores = model.predict(feature_tensors, padding_mask, candidates_to_score)
        return scores

    def _compute_loss(self, batch: SasRecTrainingBatch) -> torch.Tensor:
        if self._loss_type == "BCE":
            loss_func = self._compute_loss_bce if self._loss_sample_count is None else self._compute_loss_bce_sampled
        elif self._loss_type == "CE":
            loss_func = self._compute_loss_ce if self._loss_sample_count is None else self._compute_loss_ce_sampled
        elif self._loss_type == "SCE":
            loss_func = self._compute_loss_scalable_ce
        else:
            msg = f"Not supported loss type: {self._loss_type}"
            raise ValueError(msg)

        loss = loss_func(
            batch.features,
            batch.labels,
            batch.padding_mask,
            batch.labels_padding_mask,
        )
        return loss

    def _compute_loss_bce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # [B x L x V]
        logits = self._model.forward(feature_tensors, padding_mask)

        """
        Take only logits which correspond to non-padded tokens
        M = non_zero_count(target_padding_mask)
        """
        logits = logits[target_padding_mask]  # [M x V]
        labels = positive_labels[target_padding_mask]  # [M]

        bce_labels = torch.zeros(
            (logits.size(0), logits.size(-1)),
            device=logits.device,
        )

        # Fill positives with ones, all negatives are zeros
        bce_labels.scatter_(
            dim=-1,
            index=labels.unsqueeze(-1),
            value=1,
        )

        loss = self._loss(logits, bce_labels) / logits.size(0)
        return loss

    def _compute_loss_bce_sampled(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        (positive_logits, negative_logits, *_) = self._get_sampled_logits(
            feature_tensors, positive_labels, padding_mask, target_padding_mask
        )

        positive_prob = torch.sigmoid(positive_logits)
        negative_prob = torch.sigmoid(negative_logits)

        # Clamp and eps for numerical stability
        clamp_border: float = 100.0
        eps = 1e-6
        positive_loss = torch.clamp(torch.log((positive_prob) + eps), -clamp_border, clamp_border).sum()
        negative_loss = torch.clamp(torch.log((1 - negative_prob) + eps), -clamp_border, clamp_border).sum()

        loss = -(positive_loss + negative_loss)
        loss /= positive_logits.size(0)

        return loss

    def _compute_loss_ce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # logits: [B x L x V]
        logits = self._model.forward(
            feature_tensors,
            padding_mask,
        )

        # labels: [B x L]
        labels = positive_labels.masked_fill(mask=(~target_padding_mask), value=-100)

        logits_flat = logits.view(-1, logits.size(-1))  # [(B * L) x V]
        labels_flat = labels.view(-1)  # [(B * L)]

        loss = self._loss(logits_flat, labels_flat)
        return loss

    def _compute_loss_ce_sampled(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        assert self._loss_sample_count is not None
        (positive_logits, negative_logits, positive_labels, negative_labels, vocab_size) = self._get_sampled_logits(
            feature_tensors, positive_labels, padding_mask, target_padding_mask
        )
        n_negative_samples = min(self._loss_sample_count, vocab_size)

        # Reject negative samples matching target label & correct for remaining samples
        reject_labels = positive_labels == negative_labels  # [masked_batch_seq_size x n_negative_samples]
        negative_logits += math.log(vocab_size - 1)
        negative_logits -= 1e6 * reject_labels
        negative_logits -= torch.log((n_negative_samples - reject_labels.sum(dim=-1, keepdim=True)).float())

        # Apply regular softmax cross entropy
        # [masked_batch_seq_size x (1 + n_negative_samples)]
        logits = torch.cat([positive_logits, negative_logits], dim=1).float()
        labels_flat = torch.zeros(positive_logits.size(0), dtype=torch.long, device=padding_mask.device)
        loss = self._loss(logits, labels_flat)
        return loss

    def _compute_loss_scalable_ce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        tokens_mask: torch.BoolTensor,  # noqa: ARG002
    ) -> torch.Tensor:
        emb = self._model.forward_step(feature_tensors, padding_mask)
        all_embeddings = self.get_all_embeddings()["item_embedding"]
        return self._loss(emb, positive_labels, all_embeddings, padding_mask)

    def _get_sampled_logits(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor, int]:
        assert self._loss_sample_count is not None
        n_negative_samples = self._loss_sample_count
        positive_labels = cast(
            torch.LongTensor, torch.masked_select(positive_labels, target_padding_mask)
        )  # (masked_batch_seq_size,)
        masked_batch_seq_size = positive_labels.size(0)
        device = padding_mask.device
        output_emb = self._model.forward_step(feature_tensors, padding_mask)[target_padding_mask]

        positive_labels = cast(torch.LongTensor, positive_labels.view(-1, 1))
        ids = torch.arange(masked_batch_seq_size, dtype=torch.long, device=device)
        unique_positive_labels, positive_labels_indices = positive_labels.unique(return_inverse=True)

        if self._negative_sampling_strategy == "global_uniform":
            vocab_size = self._vocab_size
            multinomial_sample_distribution = torch.ones(vocab_size, device=device)
            # positive_labels - 2d
            positive_logits = self._model.get_logits(output_emb, positive_labels)
        elif self._negative_sampling_strategy == "inbatch":
            positive_labels_indices = positive_labels_indices.view(masked_batch_seq_size, 1)
            # unique_positive_labels - 1d
            positive_logits = self._model.get_logits(output_emb, unique_positive_labels)
            vocab_size = unique_positive_labels.size(0)
            if self._negatives_sharing:
                multinomial_sample_distribution = torch.ones(vocab_size, device=device)
            else:
                multinomial_sample_distribution = torch.softmax(positive_logits, dim=-1)
        else:
            msg = f"Unknown negative sampling strategy: {self._negative_sampling_strategy}"
            raise NotImplementedError(msg)
        n_negative_samples = min(n_negative_samples, vocab_size)

        if self._negatives_sharing:
            negative_labels = torch.multinomial(
                multinomial_sample_distribution,
                num_samples=n_negative_samples,
                replacement=False,
            )
            negative_labels = negative_labels.unsqueeze(0).repeat(masked_batch_seq_size, 1)
        elif self._negative_sampling_strategy == "global_uniform":
            negative_labels = torch.randint(
                low=0,
                high=vocab_size,
                size=(masked_batch_seq_size, n_negative_samples),
                dtype=torch.long,
                device=device,
            )
        else:
            negative_labels = torch.multinomial(
                multinomial_sample_distribution,
                num_samples=n_negative_samples,
                replacement=False,
            )
        negative_labels = cast(torch.LongTensor, negative_labels)

        if self._negative_sampling_strategy == "global_uniform":
            if self._negatives_sharing:
                unique_negative_labels, negative_labels_indices = negative_labels.unique(return_inverse=True)
                negative_labels_indices = negative_labels_indices.view(masked_batch_seq_size, n_negative_samples)
                # unique_negative_labels - 1d
                negative_logits = self._model.get_logits(output_emb, unique_negative_labels)
                negative_logits = negative_logits[ids, negative_labels_indices.T].T
            else:
                # unique_negative_labels - 1d
                negative_logits = self._model.get_logits(output_emb, negative_labels)
        else:  # self._negative_sampling_strategy == "inbatch":
            negative_labels_indices = negative_labels
            negative_logits = positive_logits
            negative_logits = negative_logits[ids, negative_labels_indices.T].T
            positive_logits = positive_logits[ids, positive_labels_indices.T].T

        return (positive_logits, negative_logits, positive_labels, negative_labels, vocab_size)

    def _create_loss(self) -> Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss]:
        if self._loss_type == "BCE":
            return torch.nn.BCEWithLogitsLoss(reduction="sum")

        if self._loss_type == "CE":
            return torch.nn.CrossEntropyLoss()

        if self._loss_type == "SCE":
            return ScalableCrossEntropyLoss(self._sce_params)

        msg = "Not supported loss_type"
        raise NotImplementedError(msg)

    def get_all_embeddings(self) -> Dict[str, torch.nn.Embedding]:
        """
        :returns: copy of all embeddings as a dictionary.
        """
        return self._model.item_embedder.get_all_embeddings()

    def set_item_embeddings_by_size(self, new_vocab_size: int):
        """
        Keep the current item embeddings and expand vocabulary with new embeddings
        initialized with xavier_normal_ for new items.

        :param new_vocab_size: Size of vocabulary with new items included.
            Must be greater then already fitted.
        """
        old_vocab_size = self._model.item_embedder.item_emb.weight.data.shape[0] - 1
        hidden_size = self._model.hidden_size

        if new_vocab_size <= old_vocab_size:
            msg = "New vocabulary size must be greater then already fitted"
            raise ValueError(msg)

        new_embedding = torch.nn.Embedding(new_vocab_size + 1, hidden_size, padding_idx=new_vocab_size)
        torch.nn.init.xavier_normal_(new_embedding.weight)
        new_embedding.weight.data[:old_vocab_size, :] = self._model.item_embedder.item_emb.weight.data[:-1, :]

        self._set_new_item_embedder_to_model(new_embedding, new_vocab_size)

    def set_item_embeddings_by_tensor(self, all_item_embeddings: torch.Tensor):
        """
        Set item embeddings with provided weights for all items.
        If new items presented, then tensor is expanded.
        The already fitted weights will be replaced with new ones.

        :param all_item_embeddings: tensor of weights for all items with
            shape (n, h), where n - number of all items, h - model hidden size.
        """
        if all_item_embeddings.dim() != 2:
            msg = "Input tensor must have (number of all items, model hidden size) shape"
            raise ValueError(msg)

        old_vocab_size = self._model.item_embedder.item_emb.weight.data.shape[0] - 1
        new_vocab_size = all_item_embeddings.shape[0]
        hidden_size = self._model.hidden_size

        if new_vocab_size < old_vocab_size:
            msg = "New vocabulary size can't be less then already fitted"
            raise ValueError(msg)
        if all_item_embeddings.shape[1] != hidden_size:
            msg = "Input tensor second dimension doesn't match model hidden size"
            raise ValueError(msg)

        new_embedding = torch.nn.Embedding(new_vocab_size + 1, hidden_size, padding_idx=new_vocab_size)
        new_embedding.weight.data[:-1, :] = all_item_embeddings

        self._set_new_item_embedder_to_model(new_embedding, new_vocab_size)

    def append_item_embeddings(self, item_embeddings: torch.Tensor):
        """
        Append provided weights for new items only to item embedder.

        :param item_embeddings: tensor of shape (n, h), where
            n - number of only new items, h - model hidden size.
        """
        if item_embeddings.dim() != 2:
            msg = "Input tensor must have (number of new items, model hidden size) shape"
            raise ValueError(msg)

        old_vocab_size = self._model.item_embedder.item_emb.weight.data.shape[0] - 1
        new_vocab_size = item_embeddings.shape[0] + old_vocab_size
        hidden_size = self._model.hidden_size

        if item_embeddings.shape[1] != hidden_size:
            msg = "Input tensor second dimension doesn't match model hidden size"
            raise ValueError(msg)

        new_embedding = torch.nn.Embedding(new_vocab_size + 1, hidden_size, padding_idx=new_vocab_size)
        new_embedding.weight.data[:old_vocab_size, :] = self._model.item_embedder.item_emb.weight.data[:-1, :]
        new_embedding.weight.data[old_vocab_size:-1, :] = item_embeddings

        self._set_new_item_embedder_to_model(new_embedding, new_vocab_size)

    @property
    def optimizer_factory(self) -> OptimizerFactory:
        """
        Returns current optimizer_factory.
        """
        return self._optimizer_factory

    @optimizer_factory.setter
    def optimizer_factory(self, optimizer_factory: OptimizerFactory) -> None:
        """
        Sets new optimizer_factory.
        :param optimizer_factory: New optimizer factory.
        """
        if isinstance(optimizer_factory, OptimizerFactory):
            self._optimizer_factory = optimizer_factory
        else:
            msg = f"Expected optimizer_factory of type OptimizerFactory, got {type(optimizer_factory)}"
            raise ValueError(msg)

    @property
    def candidates_to_score(self) -> Union[torch.LongTensor, None]:
        """
        Returns tensor of item ids to calculate scores.
        """
        return self._candidates_to_score

    @candidates_to_score.setter
    def candidates_to_score(self, candidates: Optional[torch.LongTensor] = None) -> None:
        """
        Sets tensor of item ids to calculate scores.
        :param candidates: Tensor of item ids to calculate scores.
        """
        total_item_count = self._model.item_count
        if isinstance(candidates, torch.Tensor) and candidates.dtype is torch.long:
            if 0 < candidates.shape[0] <= total_item_count:
                self._candidates_to_score = candidates
            else:
                msg = f"Expected candidates length to be between 1 and {total_item_count=}"
                raise ValueError(msg)
        elif candidates is not None:
            msg = f"Expected candidates to be of type torch.LongTensor or None, gpt {type(candidates)}"
            raise ValueError(msg)
        self._candidates_to_score = candidates

    def _set_new_item_embedder_to_model(self, new_embedding: torch.nn.Embedding, new_vocab_size: int):
        self._model.item_embedder.item_emb = new_embedding
        self._model._head._item_embedder = self._model.item_embedder
        self._vocab_size = new_vocab_size
        self._model.item_count = new_vocab_size
        self._model.padding_idx = new_vocab_size
        self._model.masking.padding_idx = new_vocab_size
        self._schema.item_id_features[self._schema.item_id_feature_name]._set_cardinality(
            new_embedding.weight.data.shape[0] - 1
        )


def _prepare_prediction_batch(
    schema: TensorSchema, max_len: int, batch: SasRecPredictionBatch
) -> SasRecPredictionBatch:
    if batch.padding_mask.shape[1] > max_len:
        msg = (
            "The length of the submitted sequence "
            "must not exceed the maximum length of the sequence. "
            f"The length of the sequence is given {batch.padding_mask.shape[1]}, "
            f"while the maximum length is {max_len}"
        )
        raise ValueError(msg)

    if batch.padding_mask.shape[1] < max_len:
        query_id, padding_mask, features = batch
        sequence_item_count = padding_mask.shape[1]
        for feature_name, feature_tensor in features.items():
            if schema[feature_name].is_cat:
                features[feature_name] = torch.nn.functional.pad(
                    feature_tensor, (max_len - sequence_item_count, 0), value=0
                )
            else:
                features[feature_name] = torch.nn.functional.pad(
                    feature_tensor.view(feature_tensor.size(0), feature_tensor.size(1)),
                    (max_len - sequence_item_count, 0),
                    value=0,
                ).unsqueeze(-1)
        padding_mask = torch.nn.functional.pad(padding_mask, (max_len - sequence_item_count, 0), value=0)
        batch = SasRecPredictionBatch(query_id, padding_mask, features)
    return batch
