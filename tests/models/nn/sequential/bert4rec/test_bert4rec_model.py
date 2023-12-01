import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.bert4rec import Bert4RecModel
    from replay.models.nn.sequential.bert4rec.model import BertEmbedding, ClassificationHead, PositionalEmbedding

torch = pytest.importorskip("torch")


@pytest.mark.torch
def test_bert_embedding_dim_mismatch(tensor_schema):
    with pytest.raises(ValueError):
        Bert4RecModel(schema=tensor_schema.subset(["item_id", "some_item_feature"]))


@pytest.mark.torch
def test_not_sequential_feature(tensor_schema):
    with pytest.raises(NotImplementedError) as exc:
        Bert4RecModel(tensor_schema.subset(["item_id", "some_user_feature"]))

    assert str(exc.value) == "Non-sequential features is not yet supported"


@pytest.mark.torch
def test_bert_embedding_not_sum_aggregation_method(tensor_schema):
    with pytest.raises(NotImplementedError):
        BertEmbedding(tensor_schema.subset(["item_id"]), aggregation_method="", max_len=100)


@pytest.mark.torch
def test_bert_embedding_cont_feature_initialize(tensor_schema):
    BertEmbedding(tensor_schema.subset(["item_id", "some_num_feature"]), max_len=100)


@pytest.mark.torch
def test_bert_embedding_shape(tensor_schema, simple_masks):
    torch.manual_seed(5555)

    item_sequences, _, tokens_mask, _ = simple_masks

    embedder = BertEmbedding(tensor_schema.subset(["item_id", "some_cat_feature"]), max_len=5)
    inputs = {
        "item_id": item_sequences,
        "some_cat_feature": item_sequences.detach().clone(),
    }

    assert embedder(inputs, tokens_mask).size() == (4, 5, 64)


@pytest.mark.torch
def test_bert_embedding_cont_features(tensor_schema, simple_masks):
    torch.manual_seed(5555)

    item_sequences, _, tokens_mask, _ = simple_masks

    embedder = BertEmbedding(tensor_schema.subset(["item_id", "some_num_feature"]), max_len=5)
    inputs = {
        "item_id": item_sequences,
        "some_num_feature": torch.rand(64),
    }

    assert embedder(inputs, tokens_mask).size() == (4, 5, 64)


@pytest.mark.torch
def test_bert_embedding_not_sum_aggregation_forward(tensor_schema):
    with pytest.raises(NotImplementedError):
        embedder = BertEmbedding(tensor_schema.subset(["item_id", "some_cat_feature"]), max_len=5)
        embedder.aggregation_method = ""
        embedder({}, torch.tensor([], dtype=torch.bool))


@pytest.mark.torch
def test_bert_positional_embedding_shape(simple_masks):
    item_sequences, _, _, _ = simple_masks

    assert PositionalEmbedding(5, 64)(item_sequences).size() == (4, 5, 64)


@pytest.mark.torch
@pytest.mark.parametrize(
    "item_ids, result_shape",
    [
        (torch.tensor([0, 1], dtype=torch.long), (4, 5, 2)),
        (torch.tensor([0], dtype=torch.long), (4, 5, 1)),
        (None, (4, 5, 4)),
    ],
)
def test_get_logits_head(tensor_schema, simple_masks, item_ids, result_shape):
    item_sequences, _, tokens_mask, _ = simple_masks

    head = ClassificationHead(64, 4)
    embedder = BertEmbedding(tensor_schema.subset(["item_id"]), max_len=5)
    inputs = {"item_id": item_sequences}

    assert head(embedder(inputs, tokens_mask), item_ids=item_ids).size() == result_shape


@pytest.mark.torch
@pytest.mark.parametrize(
    "enable_positional_embedding, enable_embedding_tying",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_dummy_bert_converges(tensor_schema, simple_masks, enable_positional_embedding, enable_embedding_tying):
    # Check if bert is able to memorize input sequences
    # We pass the same batch of sequences multiple times
    # with the last token masked and check wether BERT can
    # learn the last token embedding

    torch.manual_seed(5555)

    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    bert = Bert4RecModel(
        tensor_schema.subset(["item_id"]),
        max_len=5,
        hidden_size=64,
        enable_positional_embedding=enable_positional_embedding,
        enable_embedding_tying=enable_embedding_tying,
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    adam = torch.optim.Adam(bert.parameters(), lr=0.0001)

    input_tensors = {"item_id": item_sequences}
    labels_mask = (~padding_mask) + tokens_mask
    labels = item_sequences.masked_fill(labels_mask, -1)

    for _ in range(0, 20):
        bert.train()
        adam.zero_grad()

        scores = bert(input_tensors, padding_mask, tokens_mask)
        loss = loss_fn(scores.view(-1, scores.size(-1)), labels.view(-1))

        loss.backward()
        adam.step()

    bert.eval()

    scores = bert(input_tensors, padding_mask, tokens_mask)
    ranks = scores[:, -1, :].argsort(dim=1, descending=True)

    assert (ranks[:, 0] == torch.tensor([2, 2, 2, 2], dtype=torch.long)).all()
