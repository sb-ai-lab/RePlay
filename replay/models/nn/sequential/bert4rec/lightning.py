import math
from typing import Any, Dict, Optional, Tuple, Union, cast

import lightning
import torch

from replay.data.nn import TensorMap, TensorSchema
from replay.models.nn.optimizer_utils import FatOptimizerFactory, LRSchedulerFactory, OptimizerFactory

from .dataset import Bert4RecPredictionBatch, Bert4RecTrainingBatch, Bert4RecValidationBatch, _shift_features
from .model import Bert4RecModel, CatFeatureEmbedding


class Bert4Rec(lightning.LightningModule):
    """
    Implements BERT training-validation loop
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        block_count: int = 2,
        head_count: int = 4,
        hidden_size: int = 256,
        max_seq_len: int = 100,
        dropout_rate: float = 0.1,
        pass_per_transformer_block_count: int = 1,
        enable_positional_embedding: bool = True,
        enable_embedding_tying: bool = False,
        loss_type: str = "CE",
        loss_sample_count: Optional[int] = None,
        negative_sampling_strategy: str = "global_uniform",
        negatives_sharing: bool = False,
        optimizer_factory: Optional[OptimizerFactory] = None,
        lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
    ):
        """
        :param tensor_schema (TensorSchema): Tensor schema of features.
        :param block_count: Number of Transformer blocks.
            Default: ``2``.
        :param head_count: Number of Attention heads.
            Default: ``4``.
        :param hidden_size: Hidden size of transformer.
            Default: ``256``.
        :param max_seq_len: Max length of sequence.
            Default: ``100``.
        :param dropout_rate (float): Dropout rate.
            Default: ``0.1``.
        :param pass_per_transformer_block_count: Number of times to pass data over each Transformer block.
            Default: ``1``.
        :param enable_positional_embedding: Add positional embedding to the result.
            Default: ``True``.
        :param enable_embedding_tying: Use embedding tying head.
            If `True` - result scores are calculated by dot product of input and output embeddings,
            if `False` - default linear layer is applied to calculate logits for each item.
            Default: ``False``.
        :param loss_type: Loss type. Possible values: ``"CE"``, ``"BCE"``.
            Default: ``CE``.
        :param loss_sample_count (Optional[int]): Sample count to calculate loss.
            Default: ``None``.
        :param negative_sampling_strategy: Negative sampling strategy to calculate loss on sampled negatives.
            Is used when large count of items in dataset.
            Possible values: ``"global_uniform"``, ``"inbatch"``
            Default: ``global_uniform``.
        :param negatives_sharing: Apply negative sharing in calculating sampled logits.
            Default: ``False``.
        :param optimizer_factory: Optimizer factory.
            Default: ``None``.
        :param lr_scheduler_factory: Learning rate schedule factory.
            Default: ``None``.
        """
        super().__init__()
        self.save_hyperparameters()
        self._model = Bert4RecModel(
            schema=tensor_schema,
            max_len=max_seq_len,
            hidden_size=hidden_size,
            num_blocks=block_count,
            num_heads=head_count,
            num_passes_over_block=pass_per_transformer_block_count,
            dropout=dropout_rate,
            enable_positional_embedding=enable_positional_embedding,
            enable_embedding_tying=enable_embedding_tying,
        )
        self._loss_type = loss_type
        self._loss_sample_count = loss_sample_count
        self._negative_sampling_strategy = negative_sampling_strategy
        self._negatives_sharing = negatives_sharing
        self._optimizer_factory = optimizer_factory
        self._lr_scheduler_factory = lr_scheduler_factory
        self._loss = self._create_loss()
        self._schema = tensor_schema
        assert negative_sampling_strategy in {"global_uniform", "inbatch"}

        item_count = tensor_schema.item_id_features.item().cardinality
        assert item_count
        self._vocab_size = item_count

    def training_step(self, batch: Bert4RecTrainingBatch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """
        :param batch: Batch of training data.
        :param batch_idx: Batch index.

        :returns: Computed loss for batch.
        """
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def forward(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        tokens_mask: torch.BoolTensor,
    ) -> torch.Tensor:  # pragma: no cover
        """
        :param feature_tensors:  Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param tokens_mask: Token mask where 0 - <MASK> tokens, 1 otherwise.

        :returns: Calculated scores.
        """
        return self._model_predict(feature_tensors, padding_mask, tokens_mask)

    def predict_step(
        self, batch: Bert4RecPredictionBatch, batch_idx: int, dataloader_idx: int = 0  # noqa: ARG002
    ) -> torch.Tensor:
        """
        :param batch (Bert4RecPredictionBatch): Batch of prediction data.
        :param batch_idx (int): Batch index.
        :param dataloader_idx (int): Dataloader index.

        :returns: Calculated scores on prediction batch.
        """
        batch = self._prepare_prediction_batch(batch)
        return self._model_predict(batch.features, batch.padding_mask, batch.tokens_mask)

    def validation_step(
        self, batch: Bert4RecValidationBatch, batch_idx: int, dataloader_idx: int = 0  # noqa: ARG002
    ) -> torch.Tensor:
        """
        :param batch: Batch of prediction data.
        :param batch_idx: Batch index.

        :returns: Calculated scores on validation batch.
        """
        return self._model_predict(batch.features, batch.padding_mask, batch.tokens_mask)

    def configure_optimizers(self) -> Any:
        """
        :returns: Configured optimizer and lr scheduler.
        """
        optimizer_factory = self._optimizer_factory or FatOptimizerFactory()
        optimizer = optimizer_factory.create(self._model.parameters())

        if self._lr_scheduler_factory is None:
            return optimizer

        lr_scheduler = self._lr_scheduler_factory.create(optimizer)
        return [optimizer], [lr_scheduler]

    def _prepare_prediction_batch(self, batch: Bert4RecPredictionBatch) -> Bert4RecPredictionBatch:
        if batch.padding_mask.shape[1] > self._model.max_len:
            msg = f"The length of the submitted sequence \
                must not exceed the maximum length of the sequence. \
                The length of the sequence is given {batch.padding_mask.shape[1]}, \
                while the maximum length is {self._model.max_len}"
            raise ValueError(msg)

        if batch.padding_mask.shape[1] < self._model.max_len:
            query_id, padding_mask, features, _ = batch
            sequence_item_count = padding_mask.shape[1]
            for feature_name, feature_tensor in features.items():
                if self._schema[feature_name].is_cat:
                    features[feature_name] = torch.nn.functional.pad(
                        feature_tensor, (self._model.max_len - sequence_item_count, 0), value=0
                    )
                else:
                    features[feature_name] = torch.nn.functional.pad(
                        feature_tensor.view(feature_tensor.size(0), feature_tensor.size(1)),
                        (self._model.max_len - sequence_item_count, 0),
                        value=0,
                    ).unsqueeze(-1)
            padding_mask = torch.nn.functional.pad(
                padding_mask, (self._model.max_len - sequence_item_count, 0), value=0
            )
            shifted_features, shifted_padding_mask, tokens_mask = _shift_features(self._schema, features, padding_mask)
            batch = Bert4RecPredictionBatch(query_id, shifted_padding_mask, shifted_features, tokens_mask)
        return batch

    def _model_predict(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        tokens_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        model: Bert4RecModel
        if isinstance(self._model, torch.nn.DataParallel):
            model = cast(Bert4RecModel, self._model.module)  # multigpu
        else:
            model = self._model
        scores = model(feature_tensors, padding_mask, tokens_mask)
        candidate_scores = scores[:, -1, :]
        return candidate_scores

    def _compute_loss(self, batch: Bert4RecTrainingBatch) -> torch.Tensor:
        if self._loss_type == "BCE":
            loss_func = self._compute_loss_bce if self._loss_sample_count is None else self._compute_loss_bce_sampled
        elif self._loss_type == "CE":
            loss_func = self._compute_loss_ce if self._loss_sample_count is None else self._compute_loss_ce_sampled
        else:
            msg = f"Not supported loss type: {self._loss_type}"
            raise ValueError(msg)

        loss = loss_func(
            batch.features,
            batch.labels,
            batch.padding_mask,  # 0 - padding_idx, 1 - other tokens
            batch.tokens_mask,  # 0 - masked token, 1 - non-masked token
        )

        return loss

    def _compute_loss_bce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        tokens_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # [B x L x V]
        logits = self._model(feature_tensors, padding_mask, tokens_mask)

        labels_mask = (~padding_mask) + tokens_mask
        masked_tokens = ~labels_mask
        """
        Take only logits which correspond to non-padded tokens
        M = non_zero_count(target_padding_mask)
        """
        logits = logits[masked_tokens]  # [M x V]
        labels = positive_labels[masked_tokens]  # [M]

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
        tokens_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        (positive_logits, negative_logits, *_) = self._get_sampled_logits(
            feature_tensors, positive_labels, padding_mask, tokens_mask
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
        tokens_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # B -- batch size
        # L -- sequence length
        # V -- number of items to predict
        logits: torch.Tensor = self._model(feature_tensors, padding_mask, tokens_mask)  # [B x L x V]

        labels_mask = (~padding_mask) + tokens_mask
        masked_tokens = ~labels_mask

        logits = logits[masked_tokens]
        masked_labels = positive_labels[masked_tokens]

        loss = self._loss(logits, masked_labels)
        return loss

    def _compute_loss_ce_sampled(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        tokens_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        assert self._loss_sample_count is not None
        (positive_logits, negative_logits, positive_labels, negative_labels, vocab_size) = self._get_sampled_logits(
            feature_tensors, positive_labels, padding_mask, tokens_mask
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

    def _get_sampled_logits(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        tokens_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor, int]:
        assert self._loss_sample_count is not None
        n_negative_samples = self._loss_sample_count

        labels_mask = (~padding_mask) + tokens_mask
        masked_tokens = ~labels_mask

        positive_labels = cast(
            torch.LongTensor, torch.masked_select(positive_labels, masked_tokens)
        )  # (masked_batch_seq_size,)
        masked_batch_seq_size = positive_labels.size(0)
        device = padding_mask.device
        ids = torch.arange(masked_batch_seq_size, dtype=torch.long, device=device)
        output_emb = self._model.forward_step(feature_tensors, padding_mask, tokens_mask)[masked_tokens]

        unique_positive_labels, positive_labels_indices = positive_labels.unique(return_inverse=True)
        positive_labels_indices = positive_labels_indices.view(masked_batch_seq_size, 1)
        positive_labels = cast(torch.LongTensor, positive_labels.view(-1, 1))
        positive_logits = self._model.get_logits(output_emb, unique_positive_labels)

        if self._negative_sampling_strategy == "global_uniform":
            vocab_size = self._vocab_size
            multinomial_sample_distribution = torch.ones(vocab_size, device=device)
        elif self._negative_sampling_strategy == "inbatch":
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

        if self._negative_sampling_strategy in {"global_uniform"}:
            unique_negative_labels, negative_labels_indices = negative_labels.unique(return_inverse=True)
            negative_labels_indices = negative_labels_indices.view(masked_batch_seq_size, n_negative_samples)
            negative_logits = self._model.get_logits(output_emb, unique_negative_labels)
        else:  # inbatch
            negative_labels_indices = negative_labels
            negative_logits = positive_logits

        # [masked_batch_seq_size x 1]
        positive_logits = positive_logits[ids, positive_labels_indices.T].T
        # [masked_batch_seq_size x n_negative_samples]
        negative_logits = negative_logits[ids, negative_labels_indices.T].T
        return (
            positive_logits,
            negative_logits,
            positive_labels,
            cast(torch.LongTensor, negative_labels),
            vocab_size,
        )

    def _create_loss(self) -> Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss]:
        if self._loss_type == "BCE":
            return torch.nn.BCEWithLogitsLoss(reduction="sum")

        if self._loss_type == "CE":
            return torch.nn.CrossEntropyLoss()

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
        if new_vocab_size <= self._vocab_size:
            msg = "New vocabulary size must be greater then already fitted"
            raise ValueError(msg)

        item_tensor_feature_info = self._model.schema.item_id_features.item()
        item_tensor_feature_info._set_cardinality(new_vocab_size)

        weights_new = CatFeatureEmbedding(item_tensor_feature_info)
        torch.nn.init.xavier_normal_(weights_new.weight)
        weights_new.weight.data[: self._vocab_size, :] = self._model.item_embedder.item_embeddings.data

        self._set_new_item_embedder_to_model(weights_new, new_vocab_size)

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

        new_vocab_size = all_item_embeddings.shape[0]
        if new_vocab_size < self._vocab_size:
            msg = "New vocabulary size can't be less then already fitted"
            raise ValueError(msg)

        item_tensor_feature_info = self._model.schema.item_id_features.item()
        if all_item_embeddings.shape[1] != item_tensor_feature_info.embedding_dim:
            msg = "Input tensor second dimension doesn't match embedding dim"
            raise ValueError(msg)

        item_tensor_feature_info._set_cardinality(new_vocab_size)

        weights_new = CatFeatureEmbedding(item_tensor_feature_info)
        torch.nn.init.xavier_normal_(weights_new.weight)
        weights_new.weight.data[:new_vocab_size, :] = all_item_embeddings.data

        self._set_new_item_embedder_to_model(weights_new, new_vocab_size)

    def append_item_embeddings(self, item_embeddings: torch.Tensor):
        """
        Append provided weights for new items only to item embedder.

        :param item_embeddings: tensor of shape (n, h), where
            n - number of only new items, h - model hidden size.
        """
        if item_embeddings.dim() != 2:
            msg = "Input tensor must have (number of all items, model hidden size) shape"
            raise ValueError(msg)

        new_vocab_size = item_embeddings.shape[0] + self._vocab_size

        item_tensor_feature_info = self._model.schema.item_id_features.item()
        if item_embeddings.shape[1] != item_tensor_feature_info.embedding_dim:
            msg = "Input tensor second dimension doesn't match embedding dim"
            raise ValueError(msg)

        item_tensor_feature_info._set_cardinality(new_vocab_size)

        weights_new = CatFeatureEmbedding(item_tensor_feature_info)
        torch.nn.init.xavier_normal_(weights_new.weight)
        weights_new.weight.data[: self._vocab_size, :] = self._model.item_embedder.item_embeddings.data
        weights_new.weight.data[self._vocab_size :, :] = item_embeddings.data

        self._set_new_item_embedder_to_model(weights_new, new_vocab_size)

    def _set_new_item_embedder_to_model(self, weights_new: torch.nn.Embedding, new_vocab_size: int):
        self._model.item_embedder.cat_embeddings[self._model.schema.item_id_feature_name] = weights_new

        if self._model.enable_embedding_tying is True:
            self._model._head._item_embedder = self._model.item_embedder
            new_bias = torch.Tensor(new_vocab_size)
            new_bias.normal_(0, 0.01)
            new_bias[: self._vocab_size] = self._model._head.out_bias.data
            self._model._head.out_bias = torch.nn.Parameter(new_bias)
        else:
            new_linear = torch.nn.Linear(self._model.hidden_size, new_vocab_size)
            new_linear.weight.data[: self._vocab_size, :] = self._model._head.linear.weight.data
            new_linear.bias.data[: self._vocab_size] = self._model._head.linear.bias.data
            self._model._head.linear = new_linear

        self._vocab_size = new_vocab_size
        self._model.item_count = new_vocab_size
