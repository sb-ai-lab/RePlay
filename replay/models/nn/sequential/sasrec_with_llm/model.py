import gc
from typing import Tuple, Union

import torch
from torch import nn

from replay.data.nn import TensorMap, TensorSchema

from replay.models.nn.sequential.sasrec import SasRecModel
from replay.models.nn.sequential.sasrec.model import SasRecLayers, TiSasRecLayers
from replay.models.nn.sequential.sasrec_with_llm.utils import mean_weightening, exponential_weightening, \
    SimpleAttentionAggregator


class SasRecLLMModel(SasRecModel):
    """
    SasRec model with LLM profiles
    """

    def __init__(
        self,
        schema: TensorSchema,
        profile_emb_dim: int,
        num_blocks: int = 2,
        num_heads: int = 1,
        hidden_size: int = 50,
        max_len: int = 200,
        dropout: float = 0.2,
        ti_modification: bool = False,
        time_span: int = 256,
        reconstruction_layer: int = -1,
        weighting_scheme='mean',
        use_down_scale=True,
        use_upscale=False,
        weight_scale=None,
        multi_profile=False,
        multi_profile_aggr_scheme='mean'
    ) -> None:
        """
        :param schema: Tensor schema of features.
        :param num_blocks: Number of Transformer blocks.
            Default: ``2``.
        :param num_heads: Number of Attention heads.
            Default: ``1``.
        :param hidden_size: Hidden size of transformer.
            Default: ``50``.
        :param max_len: Max length of sequence.
            Default: ``200``.
        :param dropout: Dropout rate.
            Default: ``0.2``.
        :param ti_modification: Enable time relation.
            Default: ``False``.
        :param time_span: Time span if `ti_modification` is `True`.
            Default: ``256``.
        :param reconstruction_layer: Layer to use for reconstruction.
            Default: ``-1``.
        :param weighting_scheme: Weighting scheme for aggregation.
            Default: ``'mean'``.
        :param use_down_scale: Use downscale.
            Default: ``True``.
        :param use_upscale: Use upscale.
            Default: ``False``.
        :param weight_scale: Weight scale for exponential weighting.
            Default: ``None``.
        :param multi_profile: Use multi-profile.
            Default: ``False``.
        :param multi_profile_aggr_scheme: Multi-profile aggregation scheme.
            Default: ``'mean'``.
        """
        super().__init__(schema=schema,
                         num_blocks=num_blocks,
                         num_heads=num_heads,
                         hidden_size=hidden_size,
                         max_len=max_len,
                         dropout=dropout,
                         ti_modification=ti_modification,
                         time_span=time_span)

        # override SASRec layers to return hidden states
        del self.sasrec_layers
        gc.collect()

        if self.ti_modification:
            self.sasrec_layers = TiSasRecWithHiddenLayers(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                dropout=self.dropout,
            )
        else:
            self.sasrec_layers = SasRecWithHiddenLayers(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                dropout=self.dropout,
            )

        if weighting_scheme == 'mean':
            self.weighting_fn = mean_weightening
            self.weighting_kwargs = {}
        elif weighting_scheme == 'exponential':
            self.weighting_fn = exponential_weightening
            self.weighting_kwargs = {'weight_scale': weight_scale}
        elif weighting_scheme == 'attention':
            self.weighting_fn = SimpleAttentionAggregator(self.hidden_size)
            self.weighting_kwargs = {}
        else:
            raise NotImplementedError(f'No such weighting_scheme {weighting_scheme} exists')

        if multi_profile_aggr_scheme == 'mean':
            self.profile_aggregator = mean_weightening
            self.multi_profile_weighting_kwargs = {}
        elif multi_profile_aggr_scheme == 'attention':
            self.profile_aggregator = SimpleAttentionAggregator(profile_emb_dim if not use_down_scale
                                                                else self.hidden_size)
            self.multi_profile_weighting_kwargs = {}
        else:
            raise NotImplementedError(f'No such multi_profile_aggr_scheme {multi_profile_aggr_scheme} exists')

        self.use_down_scale = use_down_scale
        self.use_upscale = use_upscale
        self.multi_profile = multi_profile
        self.reconstruction_layer = reconstruction_layer

        if use_down_scale:
            self.profile_transform = nn.Linear(profile_emb_dim, self.hidden_size)
        if use_upscale:
            self.hidden_layer_transform = nn.Linear(self.hidden_size, profile_emb_dim)

    def forward(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Calculated scores.
        """
        output_embeddings, hidden_states = self.forward_step(feature_tensor, padding_mask)
        all_scores = self.get_logits(output_embeddings)
        return all_scores, hidden_states

    def get_query_embeddings(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ):
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Query embeddings.
        """
        output, hidden_state = self.forward_step(feature_tensor, padding_mask)
        return output[:, -1, :]

    def forward_step(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Output embeddings.
        """
        device = feature_tensor[self.item_feature_name].device
        attention_mask, padding_mask, feature_tensor = self.masking(feature_tensor, padding_mask)
        if self.ti_modification:
            seqs, ti_embeddings = self.item_embedder(feature_tensor, padding_mask)
            seqs, hidden_states__list = self.sasrec_layers(seqs, attention_mask, padding_mask, ti_embeddings, device)
        else:
            seqs = self.item_embedder(feature_tensor, padding_mask)
            seqs, hidden_states__list = self.sasrec_layers(seqs, attention_mask, padding_mask)
        output_embeddings = self.output_normalization(seqs)
        if self.reconstruction_layer == -1:
            hidden_states = output_embeddings
        else:
            hidden_states = hidden_states__list[self.reconstruction_layer]
        hidden_states_agg = self.weighting_fn(hidden_states, **self.weighting_kwargs)
        return output_embeddings, hidden_states_agg

    def aggregate_profile(self, user_profile_emb):
        """
        :param user_profile_emb: User profile embeddings.

        :returns: Aggregated user profile embeddings.
        """
        if user_profile_emb is None:
            return None

        # single-profile [batch_size, emb_dim]
        if user_profile_emb.dim() == 2:
            if self.use_down_scale:
                return self.profile_transform(user_profile_emb)
            else:
                return user_profile_emb.detach().clone()

        # multi-profile [batch_size, K, emb_dim]
        bsz, K, edim = user_profile_emb.shape

        if self.use_down_scale:
            user_profile_emb = user_profile_emb.view(bsz * K, edim)
            user_profile_emb = self.profile_transform(user_profile_emb)
            user_profile_emb = user_profile_emb.view(bsz, K, self.hidden_size)

        aggregated = self.profile_aggregator(user_profile_emb,
                                             *self.multi_profile_weighting_kwargs)
        return aggregated


class SasRecWithHiddenLayers(SasRecLayers):
    """
    SasRec vanilla layers with hidden states:
        1. SelfAttention layers
        2. FeedForward layers

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param num_heads: Number of Attention heads.
        :param num_blocks: Number of Transformer blocks.
        :param dropout: Dropout rate.
        """
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         num_blocks=num_blocks,
                         dropout=dropout)

    def forward(
        self,
        seqs: torch.Tensor,
        attention_mask: torch.BoolTensor,
        padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, list]:
        """
        :param seqs: Item embeddings.
        :param attention_mask: Attention mask.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Output embeddings.
        """
        hidden_states__list = []
        num_blocks = len(self.attention_layers)
        for i in range(num_blocks):
            query = self.attention_layernorms[i](seqs)
            attent_emb, _ = self.attention_layers[i](
                query, seqs, seqs, attn_mask=attention_mask, need_weights=False)
            seqs = query + attent_emb

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= padding_mask

            hidden_states__list.append(seqs.clone())
        return seqs, hidden_states__list


class TiSasRecWithHiddenLayers(TiSasRecLayers):
    """
    TiSasRec layers with hidden states:
        1. Time-relative SelfAttention layers
        2. FeedForward layers

    Link: https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param num_heads: Number of Attention heads.
        :param num_blocks: Number of Transformer blocks.
        :param dropout: Dropout rate.
        """
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         num_blocks=num_blocks,
                         dropout=dropout)

    def forward(
        self,
        seqs: torch.Tensor,
        attention_mask: torch.BoolTensor,
        padding_mask: torch.BoolTensor,
        ti_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """
        :param seqs: Item embeddings.
        :param attention_mask: Attention mask.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param ti_embeddings: Output embeddings, key and value time interval matrices,
            key and value positional embeddings.
        :param device: Selected device.

        :returns: Output embeddings.
        """
        hidden_states = []
        length = len(self.attention_layers)
        for i in range(length):
            query = self.attention_layernorms[i](seqs)
            attent_emb = self.attention_layers[i](query, seqs, ~padding_mask, attention_mask, ti_embeddings, device)
            seqs = query + attent_emb
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= padding_mask

            hidden_states.append(seqs.clone())
        return seqs, hidden_states
