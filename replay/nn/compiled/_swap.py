"""Inference swap: EmbeddingBag -> Embedding + reduction (D7)."""

import torch


class ExportableEmbeddingBag(torch.nn.Module):
    """ONNX-exportable equivalent of ``torch.nn.EmbeddingBag``.

    Reproduces the bag reduction with an explicit reduction so the graph
    contains no bag ops. The weight ``Parameter`` is tied to the original
    bag (no copy); ``padding_idx`` entries count toward no mode.
    """

    def __init__(self, bag: torch.nn.EmbeddingBag):
        super().__init__()
        self._mode = bag.mode
        emb = torch.nn.Embedding(bag.num_embeddings, bag.embedding_dim, padding_idx=bag.padding_idx)
        emb.weight = bag.weight
        self.emb = emb

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def padding_idx(self) -> int | None:
        return self.emb.padding_idx

    @property
    def weight(self) -> torch.Tensor:
        return self.emb.weight

    @property
    def num_embeddings(self) -> int:
        return self.emb.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.emb.embedding_dim

    def _fill_padding_idx_with_zero(self) -> None:
        if self.emb.padding_idx is not None:
            with torch.no_grad():
                self.emb.weight[self.emb.padding_idx].fill_(0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embedded = self.emb(indices)
        if self._mode == "sum":
            return embedded.sum(dim=1)

        valid = indices != self.emb.padding_idx
        count = valid.sum(dim=1, keepdim=True)
        if self._mode == "mean":
            return embedded.sum(dim=1) / count.clamp_min(1)

        neg_inf = torch.finfo(embedded.dtype).min
        masked = embedded.masked_fill(~valid.unsqueeze(-1), neg_inf)
        maximum = masked.max(dim=1).values
        return torch.where(valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))


def swap_embedding_bags(module: torch.nn.Module) -> int:
    """Recursively replace ``EmbeddingBag`` submodules in-place.

    Weights are tied, so the module's outputs are unchanged; only the
    graph representation becomes ONNX-exportable.

    :param module: the module tree to prepare for export.
    :return: number of replaced bags.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.EmbeddingBag):
            setattr(module, name, ExportableEmbeddingBag(child))
            replaced += 1
        else:
            replaced += swap_embedding_bags(child)
    return replaced
