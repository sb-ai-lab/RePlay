# import numpy as np
# import pandas as pd
# import pytest
# import torch
# from replay.nn.lightning.postprocessors import SeenItemsFilter


# @pytest.mark.torch
# @pytest.mark.parametrize(
#     "dataset_type",
#     [
#         pytest.param("item_user_sequential_dataset"),
#         pytest.param("polars_item_user_sequential_dataset"),
#     ],
# )
# @pytest.mark.parametrize(
#     "query_ids, scores, unseen_items",
#     [
#         (
#             torch.tensor([0], dtype=torch.long),
#             torch.tensor([0, 1, 2, 3, 4], dtype=torch.float),
#             torch.tensor([False, False, False, True, True], dtype=torch.bool),
#         ),
#         (
#             torch.tensor([1], dtype=torch.long),
#             torch.tensor([0, 1, 2, 3, 4], dtype=torch.float),
#             torch.tensor([False, False, False, False, True], dtype=torch.bool),
#         ),
#         (
#             torch.tensor([2, 3], dtype=torch.long),
#             torch.tensor([[0, 1, 2, 3, 4], [2, 4, 1, 3, 3]], dtype=torch.float),
#             torch.tensor([[False, False, False, False, True], [True, False, False, False, True]], dtype=torch.bool),
#         ),
#     ],
# )
# def test_remove_seen_items_on_predict(dataset_type, query_ids, scores, unseen_items, request):
#     test_dataframe = request.getfixturevalue(dataset_type)
#     postprocessor = SeenItemsFilter(test_dataframe)
#     _, scores_pred = postprocessor.on_prediction(query_ids=query_ids, scores=scores)

#     scores_pred_unseen = scores_pred.flatten() > -np.inf

#     assert all(scores_pred_unseen == unseen_items.flatten())


# @pytest.mark.torch
# @pytest.mark.parametrize(
#     "query_ids, scores, unseen_items",
#     [
#         (
#             torch.tensor([0], dtype=torch.long),
#             torch.tensor([0, 1, 2, 3, 4], dtype=torch.float),
#             torch.tensor([False, False, False, True, True], dtype=torch.bool),
#         ),
#         (
#             torch.tensor([1], dtype=torch.long),
#             torch.tensor([0, 1, 2, 3, 4], dtype=torch.float),
#             torch.tensor([False, False, False, False, True], dtype=torch.bool),
#         ),
#         (
#             torch.tensor([2, 3], dtype=torch.long),
#             torch.tensor([[0, 1, 2, 3, 4], [2, 4, 1, 3, 3]], dtype=torch.float),
#             torch.tensor([[False, False, False, False, True], [True, False, False, False, True]], dtype=torch.bool),
#         ),
#     ],
# )
# def test_remove_seen_items_on_validation(item_user_sequential_dataset, query_ids, scores, unseen_items):
#     postprocessor = SeenItemsFilter(item_user_sequential_dataset)
#     _, scores_val, _ = postprocessor.on_validation(query_ids=query_ids, scores=scores, ground_truth=torch.tensor([]))
#     _, scores_val, _ = postprocessor.on_validation(query_ids=query_ids, scores=scores, ground_truth=torch.tensor([]))

#     scores_val_unseen = scores_val.flatten() > -np.inf

#     assert all(scores_val_unseen == unseen_items.flatten())


# @pytest.mark.torch
# def test_not_contiguous_scores(item_user_sequential_dataset):
#     postprocessor = SeenItemsFilter(item_user_sequential_dataset)

#     scores = torch.tensor([[0, 1, 2, 3, 4], [2, 4, 1, 3, 3]], dtype=torch.float).transpose(0, 1)
#     result = torch.tensor([[False, False, False, False, True], [True, False, False, False, True]], dtype=torch.bool)

#     _, scores_pred = postprocessor.on_prediction(query_ids=torch.tensor([2, 3]), scores=scores)

#     assert all((scores_pred.flatten() > -np.inf) == result.flatten())
