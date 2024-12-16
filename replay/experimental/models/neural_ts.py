import os
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output
from pyspark.sql import DataFrame
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from replay.experimental.models.base_rec import HybridRecommender
from replay.splitters import TimeSplitter
from replay.utils.spark_utils import convert2spark

pd.options.mode.chained_assignment = None


def cartesian_product(left, right):
    """
    This function computes cartesian product.
    """
    return left.assign(key=1).merge(right.assign(key=1), on="key").drop(columns=["key"])


def num_tries_gt_zero(scores, batch_size, max_trials, max_num, device):
    """
    scores: [batch_size x N] float scores
    returns: [batch_size x 1] the lowest indice per row where scores were first greater than 0. plus 1
    """
    tmp = scores.gt(0).nonzero().t()
    # We offset these values by 1 to look for unset values (zeros) later
    values = tmp[1] + 1
    # Sparse tensors can't be moved with .to() or .cuda() if you want to send in cuda variables first
    if device.type == "cuda":
        tau = torch.cuda.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials + 1))).to_dense()
    else:
        tau = torch.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials + 1))).to_dense()
    tau[(tau == 0)] += max_num  # set all unused indices to be max possible number so its not picked by min() call

    tries = torch.min(tau, dim=1)[0]
    return tries


def w_log_loss(output, target, device):
    """
    This function computes weighted logistic loss.
    """
    output = torch.nn.functional.sigmoid(output)
    output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
    count_1 = target.sum().item()
    count_0 = target.shape[0] - count_1
    class_count = np.array([count_0, count_1])
    if count_1 == 0 or count_0 == 0:  # noqa: SIM108
        weight = np.array([1.0, 1.0])
    else:
        weight = np.max(class_count) / class_count
    weight = Tensor(weight).to(device)
    loss = weight[1] * target * torch.log(output) + weight[0] * (1 - target) * torch.log(1 - output)
    return -loss.mean()


def warp_loss(positive_predictions, negative_predictions, num_labels, device):
    """
    positive_predictions: [batch_size x 1] floats between -1 to 1
    negative_predictions: [batch_size x N] floats between -1 to 1
    num_labels: int total number of labels in dataset (not just the subset you're using for the batch)
    device: pytorch.device
    """
    batch_size, max_trials = negative_predictions.size(0), negative_predictions.size(1)

    offsets, ones, max_num = (
        torch.arange(0, batch_size, 1).long().to(device) * (max_trials + 1),
        torch.ones(batch_size, 1).float().to(device),
        batch_size * (max_trials + 1),
    )

    sample_scores = 1 + negative_predictions - positive_predictions
    # Add column of ones so we know when we used all our attempts.
    # This is used for indexing and computing should_count_loss if no real value is above 0
    sample_scores, negative_predictions = (
        torch.cat([sample_scores, ones], dim=1),
        torch.cat([negative_predictions, ones], dim=1),
    )

    tries = num_tries_gt_zero(sample_scores, batch_size, max_trials, max_num, device)
    attempts, trial_offset = tries.float(), (tries - 1) + offsets
    # Don't count loss if we used max number of attempts
    loss_weights = torch.log(torch.floor((num_labels - 1) / attempts))
    should_count_loss = (attempts <= max_trials).float()
    losses = (
        loss_weights
        * ((1 - positive_predictions.view(-1)) + negative_predictions.view(-1)[trial_offset])
        * should_count_loss
    )

    return losses.sum()


class SamplerWithReset(SequentialSampler):
    """
    Sampler class for train dataloader.
    """

    def __iter__(self):
        self.data_source.reset()
        return super().__iter__()


class UserDatasetWithReset(Dataset):
    """
    Dataset class that takes data for a single user and
    column names for continuous data, categorical data and data for
    Wide model as well as the name of the target column.
    The class also supports sampling of negative examples.
    """

    def __init__(
        self,
        idx,
        log_train,
        user_features,
        item_features,
        list_items,
        union_cols,
        cnt_neg_samples,
        device,
        target: Optional[str] = None,
    ):
        if cnt_neg_samples is not None:
            self.cnt_neg_samples = cnt_neg_samples
            self.user_features = user_features
            self.item_features = item_features
            item_idx_user = log_train["item_idx"].values.tolist()
            self.item_idx_not_user = list(set(list_items).difference(set(item_idx_user)))
        else:
            self.cnt_neg_samples = cnt_neg_samples
            self.user_features = None
            self.item_features = None
            self.item_idx_not_user = None
        self.device = device
        self.union_cols = union_cols
        dataframe = log_train.merge(user_features, on="user_idx", how="inner")
        self.dataframe = dataframe.merge(item_features, on="item_idx", how="inner")
        self.user_idx = idx
        self.data_sample = None
        self.wide_part = Tensor(self.dataframe[self.union_cols["wide_cols"]].to_numpy().astype("float32")).to(
            self.device
        )
        self.continuous_part = Tensor(
            self.dataframe[self.union_cols["continuous_cols"]].to_numpy().astype("float32")
        ).to(self.device)
        self.cat_part = Tensor(self.dataframe[self.union_cols["cat_embed_cols"]].to_numpy().astype("float32")).to(
            self.device
        )
        self.users = Tensor(self.dataframe[["user_idx"]].to_numpy().astype("int")).to(torch.long).to(self.device)
        self.items = Tensor(self.dataframe[["item_idx"]].to_numpy().astype("int")).to(torch.long).to(self.device)
        if target is not None:
            self.target = Tensor(dataframe[target].to_numpy().astype("int")).to(self.device)
        else:
            self.target = target
        self.target_column = target

    def get_parts(self, data_sample):
        """
        Dataset method that selects user index, item indexes, categorical data,
        continuous data, data for wide model, and target value.
        """
        self.wide_part = Tensor(data_sample[self.union_cols["wide_cols"]].to_numpy().astype("float32")).to(self.device)
        self.continuous_part = Tensor(data_sample[self.union_cols["continuous_cols"]].to_numpy().astype("float32")).to(
            self.device
        )
        self.cat_part = Tensor(data_sample[self.union_cols["cat_embed_cols"]].to_numpy().astype("float32")).to(
            self.device
        )
        self.users = Tensor(data_sample[["user_idx"]].to_numpy().astype("int")).to(torch.long).to(self.device)
        self.items = Tensor(data_sample[["item_idx"]].to_numpy().astype("int")).to(torch.long).to(self.device)
        if self.target_column is not None:
            self.target = Tensor(data_sample[self.target_column].to_numpy().astype("int")).to(self.device)
        else:
            self.target = self.target_column

    def __getitem__(self, idx):
        target = -1
        if self.target is not None:
            target = self.target[idx]
        return (
            self.wide_part[idx],
            self.continuous_part[idx],
            self.cat_part[idx],
            self.users[idx],
            self.items[idx],
            target,
        )

    def __len__(self):
        if self.data_sample is not None:
            return self.data_sample.shape[0]
        else:
            return self.dataframe.shape[0]

    def get_size_features(self):
        """
        Dataset method that gets the size of features after encoding/scaling.
        """
        return self.wide_part.shape[1], self.continuous_part.shape[1], self.cat_part.shape[1]

    def reset(self):
        """
        Dataset methos that samples new negative examples..
        """
        n_samples = min(len(self.item_idx_not_user), self.cnt_neg_samples)
        if n_samples > 0:
            sample_item = np.random.choice(self.item_idx_not_user, n_samples, replace=False)
            sample_item_feat = self.item_features.loc[self.item_features["item_idx"].isin(sample_item)]
            sample_item_feat = sample_item_feat.set_axis(range(sample_item_feat.shape[0]), axis="index")
            df_sample = cartesian_product(
                self.user_features.loc[self.user_features["user_idx"] == self.user_idx], sample_item_feat
            )
            df_sample[self.target_column] = 0
            self.data_sample = pd.concat([self.dataframe, df_sample], axis=0, ignore_index=True)
            self.get_parts(self.data_sample)


class Wide(nn.Module):
    """
    Wide model based on https://arxiv.org/abs/1606.07792
    """

    def __init__(self, input_dim: int, out_dim: int = 1):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.BatchNorm1d(out_dim))
        self.out_dim = out_dim

    def forward(self, input_data):
        """
        :param input_data: wide features
        :return: torch tensor with shape batch_size*out_dim
        """
        output = self.linear(input_data)
        return output


class Deep(nn.Module):
    """
    Deep model based on https://arxiv.org/abs/1606.07792
    """

    def __init__(self, input_dim: int, out_dim: int, hidden_layers: List[int], deep_dropout: float):
        super().__init__()
        model = []
        last_size = input_dim
        for cur_size in hidden_layers:
            model += [nn.Linear(last_size, cur_size)]
            model += [nn.ReLU()]
            model += [nn.BatchNorm1d(cur_size)]
            model += [nn.Dropout(deep_dropout)]
            last_size = cur_size
        model += [nn.Linear(last_size, out_dim)]
        model += [nn.ReLU()]
        model += [nn.BatchNorm1d(out_dim)]
        model += [nn.Dropout(deep_dropout)]
        self.deep_model = nn.Sequential(*model)

    def forward(self, input_data):
        """
        :param input_data: deep features
        :return: torch tensor with shape batch_size*out_dim
        """
        output = self.deep_model(input_data)
        return output


class EmbedModel(nn.Module):
    """
    Model for getting embeddings for user and item indexes.
    """

    def __init__(self, cnt_users: int, cnt_items: int, user_embed: int, item_embed: int, crossed_embed: int):
        super().__init__()
        self.user_embed = nn.Embedding(num_embeddings=cnt_users, embedding_dim=user_embed)
        self.item_embed = nn.Embedding(num_embeddings=cnt_items, embedding_dim=item_embed)
        self.user_crossed_embed = nn.Embedding(num_embeddings=cnt_users, embedding_dim=crossed_embed)
        self.item_crossed_embed = nn.Embedding(num_embeddings=cnt_items, embedding_dim=crossed_embed)

    def forward(self, users, items):
        """
        :param users: user indexes
        :param items: item indexes
        :return: torch tensors: embedings for users, embedings for items,
        embedings for users for wide model,
        embedings for items for wide model,
        embedings for pairs (users, items) for wide model
        """
        users_to_embed = self.user_embed(users).squeeze()
        items_to_embed = self.item_embed(items).squeeze()
        cross_users = self.user_crossed_embed(users).squeeze()
        cross_items = self.item_crossed_embed(items).squeeze()
        cross = (cross_users * cross_items).sum(dim=-1).unsqueeze(-1)
        return users_to_embed, items_to_embed, cross_users, cross_items, cross


class WideDeep(nn.Module):
    """
    Wide&Deep model based on https://arxiv.org/abs/1606.07792
    """

    def __init__(
        self,
        dim_head: int,
        deep_out_dim: int,
        hidden_layers: List[int],
        size_wide_features: int,
        size_continuous_features: int,
        size_cat_features: int,
        wide_out_dim: int,
        head_dropout: float,
        deep_dropout: float,
        cnt_users: int,
        cnt_items: int,
        user_embed: int,
        item_embed: int,
        crossed_embed: int,
    ):
        super().__init__()
        self.embed_model = EmbedModel(cnt_users, cnt_items, user_embed, item_embed, crossed_embed)
        self.wide = Wide(size_wide_features + crossed_embed * 2 + 1, wide_out_dim)
        self.deep = Deep(
            size_cat_features + size_continuous_features + user_embed + item_embed,
            deep_out_dim,
            hidden_layers,
            deep_dropout,
        )
        self.head_model = nn.Sequential(nn.Linear(wide_out_dim + deep_out_dim, dim_head), nn.ReLU())
        self.last_layer = nn.Sequential(nn.Linear(dim_head, 1))
        self.head_dropout = head_dropout

    def forward_for_predict(self, wide_part, continuous_part, cat_part, users, items):
        """
        Forward method without last layer and dropout that is used for prediction.
        """
        users_to_embed, items_to_embed, cross_users, cross_items, cross = self.embed_model(users, items)
        input_deep = torch.cat((cat_part, continuous_part, users_to_embed, items_to_embed), dim=-1).squeeze()
        out_deep = self.deep(input_deep)
        wide_part = torch.cat((wide_part, cross_users, cross_items, cross), dim=-1)
        out_wide = self.wide(wide_part)
        input_data = torch.cat((out_wide, out_deep), dim=-1)
        output = self.head_model(input_data)
        return output

    def forward_dropout(self, input_data):
        """
        Forward method for multiple prediction with active dropout
        :param input_data: output of forward_for_predict
        :return: torch tensor after dropout and last linear layer
        """
        output = nn.functional.dropout(input_data, p=self.head_dropout, training=True)
        output = self.last_layer(output)
        return output

    def forward_for_embeddings(
        self, wide_part, continuous_part, cat_part, users_to_embed, items_to_embed, cross_users, cross_items, cross
    ):
        """
        Forward method after getting emdeddings for users and items.
        """
        input_deep = torch.cat((cat_part, continuous_part, users_to_embed, items_to_embed), dim=-1).squeeze()
        out_deep = self.deep(input_deep)
        wide_part = torch.cat((wide_part, cross_users, cross_items, cross), dim=-1)
        out_wide = self.wide(wide_part)
        input_data = torch.cat((out_wide, out_deep), dim=-1)
        output = self.head_model(input_data)
        output = nn.functional.dropout(output, p=self.head_dropout, training=True)
        output = self.last_layer(output)
        return output

    def forward(self, wide_part, continuous_part, cat_part, users, items):
        """
        :param wide_part: features for wide model
        :param continuous_part: continuous features
        :param cat_part: torch categorical features
        :param users: user indexes
        :param items: item indexes
        :return: relevances for pair (users, items)

        """
        users_to_embed, items_to_embed, cross_users, cross_items, cross = self.embed_model(users, items)
        output = self.forward_for_embeddings(
            wide_part, continuous_part, cat_part, users_to_embed, items_to_embed, cross_users, cross_items, cross
        )
        return output


class NeuralTS(HybridRecommender):
    """
    'Neural Thompson sampling recommender
    <https://dl.acm.org/doi/pdf/10.1145/3383313.3412214>`_  based on `Wide&Deep model
    <https://arxiv.org/abs/1606.07792>`_.

    :param user_cols: user_cols = {'continuous_cols':List[str], 'cat_embed_cols':List[str], 'wide_cols': List[str]},
        where List[str] -- some column names from user_features dataframe, which is input to the fit method,
        or empty List
    :param item_cols: item_cols = {'continuous_cols':List[str], 'cat_embed_cols':List[str], 'wide_cols': List[str]},
        where List[str] -- some column names from item_features dataframe, which is input to the fit method,
        or empty List
    :param embedding_sizes: list of length three in which
        embedding_sizes[0] = embedding size for users,
        embedding_sizes[1] = embedding size for items,
        embedding_sizes[2] = embedding size for pair (users, items)
    :param hidden_layers: list of hidden layer sizes for Deep model
    :param wide_out_dim: output size for the Wide model
    :param deep_out_dim: output size for the Deep model
    :param head_dropout: probability of an element to be zeroed for WideDeep model head
    :param deep_dropout: probability of an element to be zeroed for Deep model
    :param dim_head: output size for WideDeep model head
    :param n_epochs: number of epochs for model training
    :param opt_lr: learning rate for the AdamW optimizer
    :param lr_min: minimum learning rate value for the CosineAnnealingLR learning rate scheduler
    :param use_gpu: if true, the model will be trained on the GPU
    :param use_warp_loss: if true, then warp loss will be used otherwise weighted logistic loss.
    :param cnt_neg_samples: number of additional negative examples for each user
    :param cnt_samples_for_predict: number of sampled predictions for one user,
        which are used to estimate the mean and variance of relevance
    :param exploration_coef:  exploration coefficient
    :param plot_dir: file name where the training graphs will be saved, if None, the graphs will not be saved
    :param cnt_users: number of users, used in Wide&Deep model initialization
    :param cnt_items: number of items, used in Wide&Deep model initialization

    """

    def __init__(
        self,
        user_cols: Dict[str, List[str]] = {"continuous_cols": [], "cat_embed_cols": [], "wide_cols": []},
        item_cols: Dict[str, List[str]] = {"continuous_cols": [], "cat_embed_cols": [], "wide_cols": []},
        embedding_sizes: List[int] = [32, 32, 64],
        hidden_layers: List[int] = [32, 20],
        wide_out_dim: int = 1,
        deep_out_dim: int = 20,
        head_dropout: float = 0.8,
        deep_dropout: float = 0.4,
        dim_head: int = 20,
        n_epochs: int = 2,
        opt_lr: float = 3e-4,
        lr_min: float = 1e-5,
        use_gpu: bool = False,
        use_warp_loss: bool = True,
        cnt_neg_samples: int = 100,
        cnt_samples_for_predict: int = 10,
        exploration_coef: float = 1.0,
        cnt_users: Optional[int] = None,
        cnt_items: Optional[int] = None,
        plot_dir: Optional[str] = None,
    ):
        self.user_cols = user_cols
        self.item_cols = item_cols
        self.embedding_sizes = embedding_sizes
        self.hidden_layers = hidden_layers
        self.wide_out_dim = wide_out_dim
        self.deep_out_dim = deep_out_dim
        self.head_dropout = head_dropout
        self.deep_dropout = deep_dropout
        self.dim_head = dim_head
        self.n_epochs = n_epochs
        self.opt_lr = opt_lr
        self.lr_min = lr_min
        self.device = torch.device("cpu")
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_warp_loss = use_warp_loss
        self.cnt_neg_samples = cnt_neg_samples
        self.cnt_samples_for_predict = cnt_samples_for_predict
        self.exploration_coef = exploration_coef
        self.cnt_users = cnt_users
        self.cnt_items = cnt_items
        self.plot_dir = plot_dir

        self.size_wide_features = None
        self.size_continuous_features = None
        self.size_cat_features = None
        self.scaler_user = None
        self.encoder_intersept_user = None
        self.encoder_diff_user = None
        self.scaler_item = None
        self.encoder_intersept_item = None
        self.encoder_diff_item = None
        self.union_cols = None
        self.num_of_train_labels = None
        self.dict_true_items_val = None
        self.lr_scheduler = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def preprocess_features_fit(self, train, item_features, user_features):
        """
        This function initializes all ecoders for the features.
        """
        train_users = user_features.loc[user_features["user_idx"].isin(train["user_idx"].values.tolist())]
        wide_cols_cat = list(set(self.user_cols["cat_embed_cols"]) & set(self.user_cols["wide_cols"]))
        cat_embed_cols_not_wide = list(set(self.user_cols["cat_embed_cols"]).difference(set(wide_cols_cat)))
        if len(self.user_cols["continuous_cols"]) != 0:
            self.scaler_user = MinMaxScaler()
            self.scaler_user.fit(train_users[self.user_cols["continuous_cols"]])
        else:
            self.scaler_user = None
        if len(wide_cols_cat) != 0:
            self.encoder_intersept_user = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.encoder_intersept_user.fit(train_users[wide_cols_cat])
        else:
            self.encoder_intersept_user = None
        if len(cat_embed_cols_not_wide) != 0:
            self.encoder_diff_user = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.encoder_diff_user.fit(train_users[cat_embed_cols_not_wide])
        else:
            self.encoder_diff_user = None
        train_items = item_features.loc[item_features["item_idx"].isin(train["item_idx"].values.tolist())]
        wide_cols_cat = list(set(self.item_cols["cat_embed_cols"]) & set(self.item_cols["wide_cols"]))
        cat_embed_cols_not_wide = list(set(self.item_cols["cat_embed_cols"]).difference(set(wide_cols_cat)))
        if len(self.item_cols["continuous_cols"]) != 0:
            self.scaler_item = MinMaxScaler()
            self.scaler_item.fit(train_items[self.item_cols["continuous_cols"]])
        else:
            self.scaler_item = None
        if len(wide_cols_cat) != 0:
            self.encoder_intersept_item = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.encoder_intersept_item.fit(train_items[wide_cols_cat])
        else:
            self.encoder_intersept_item = None
        if len(cat_embed_cols_not_wide) != 0:
            self.encoder_diff_item = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.encoder_diff_item.fit(train_items[cat_embed_cols_not_wide])
        else:
            self.encoder_diff_item = None

    def preprocess_features_transform(self, item_features, user_features):
        """
        This function performs the transformation for all features.
        """
        self.union_cols = {"continuous_cols": [], "cat_embed_cols": [], "wide_cols": []}
        wide_cols_cat = list(set(self.user_cols["cat_embed_cols"]) & set(self.user_cols["wide_cols"]))
        cat_embed_cols_not_wide = list(set(self.user_cols["cat_embed_cols"]).difference(set(wide_cols_cat)))
        if len(self.user_cols["continuous_cols"]) != 0:
            users_continuous = pd.DataFrame(
                self.scaler_user.transform(user_features[self.user_cols["continuous_cols"]]),
                columns=self.user_cols["continuous_cols"],
            )
            self.union_cols["continuous_cols"] += self.user_cols["continuous_cols"]
        else:
            users_continuous = user_features[[]]
        if len(wide_cols_cat) != 0:
            users_wide_cat = pd.DataFrame(
                self.encoder_intersept_user.transform(user_features[wide_cols_cat]),
                columns=list(self.encoder_intersept_user.get_feature_names_out(wide_cols_cat)),
            )
            self.union_cols["cat_embed_cols"] += list(self.encoder_intersept_user.get_feature_names_out(wide_cols_cat))
            self.union_cols["wide_cols"] += list(
                set(self.user_cols["wide_cols"]).difference(set(self.user_cols["cat_embed_cols"]))
            ) + list(self.encoder_intersept_user.get_feature_names_out(wide_cols_cat))
        else:
            users_wide_cat = user_features[[]]
        if len(cat_embed_cols_not_wide) != 0:
            users_cat = pd.DataFrame(
                self.encoder_diff_user.transform(user_features[cat_embed_cols_not_wide]),
                columns=list(self.encoder_diff_user.get_feature_names_out(cat_embed_cols_not_wide)),
            )
            self.union_cols["cat_embed_cols"] += list(
                self.encoder_diff_user.get_feature_names_out(cat_embed_cols_not_wide)
            )
        else:
            users_cat = user_features[[]]

        transform_user_features = pd.concat(
            [user_features[["user_idx"]], users_continuous, users_wide_cat, users_cat], axis=1
        )

        wide_cols_cat = list(set(self.item_cols["cat_embed_cols"]) & set(self.item_cols["wide_cols"]))
        cat_embed_cols_not_wide = list(set(self.item_cols["cat_embed_cols"]).difference(set(wide_cols_cat)))
        if len(self.item_cols["continuous_cols"]) != 0:
            items_continuous = pd.DataFrame(
                self.scaler_item.transform(item_features[self.item_cols["continuous_cols"]]),
                columns=self.item_cols["continuous_cols"],
            )
            self.union_cols["continuous_cols"] += self.item_cols["continuous_cols"]
        else:
            items_continuous = item_features[[]]
        if len(wide_cols_cat) != 0:
            items_wide_cat = pd.DataFrame(
                self.encoder_intersept_item.transform(item_features[wide_cols_cat]),
                columns=list(self.encoder_intersept_item.get_feature_names_out(wide_cols_cat)),
            )
            self.union_cols["cat_embed_cols"] += list(self.encoder_intersept_item.get_feature_names_out(wide_cols_cat))
            self.union_cols["wide_cols"] += list(
                set(self.item_cols["wide_cols"]).difference(set(self.item_cols["cat_embed_cols"]))
            ) + list(self.encoder_intersept_item.get_feature_names_out(wide_cols_cat))
        else:
            items_wide_cat = item_features[[]]
        if len(cat_embed_cols_not_wide) != 0:
            items_cat = pd.DataFrame(
                self.encoder_diff_item.transform(item_features[cat_embed_cols_not_wide]),
                columns=list(self.encoder_diff_item.get_feature_names_out(cat_embed_cols_not_wide)),
            )
            self.union_cols["cat_embed_cols"] += list(
                self.encoder_diff_item.get_feature_names_out(cat_embed_cols_not_wide)
            )
        else:
            items_cat = item_features[[]]

        transform_item_features = pd.concat(
            [item_features[["item_idx"]], items_continuous, items_wide_cat, items_cat], axis=1
        )
        return transform_user_features, transform_item_features

    def _data_loader(
        self, idx, log_train, transform_user_features, transform_item_features, list_items, train=False
    ) -> Union[Tuple[UserDatasetWithReset, DataLoader], DataLoader]:
        if train:
            train_dataset = UserDatasetWithReset(
                idx=idx,
                log_train=log_train,
                user_features=transform_user_features,
                item_features=transform_item_features,
                list_items=list_items,
                union_cols=self.union_cols,
                cnt_neg_samples=self.cnt_neg_samples,
                device=self.device,
                target="relevance",
            )
            sampler = SamplerWithReset(train_dataset)
            train_dataloader = DataLoader(
                train_dataset, batch_size=log_train.shape[0] + self.cnt_neg_samples, sampler=sampler
            )
            return train_dataset, train_dataloader
        else:
            dataset = UserDatasetWithReset(
                idx=idx,
                log_train=log_train,
                user_features=transform_user_features,
                item_features=transform_item_features,
                list_items=list_items,
                union_cols=self.union_cols,
                cnt_neg_samples=None,
                device=self.device,
                target=None,
            )
            dataloader = DataLoader(dataset, batch_size=log_train.shape[0], shuffle=False)
            return dataloader

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if user_features is None:
            msg = "User features are missing for fitting"
            raise ValueError(msg)
        if item_features is None:
            msg = "Item features are missing for fitting"
            raise ValueError(msg)

        train_spl = TimeSplitter(
            time_threshold=0.2,
            drop_cold_items=True,
            drop_cold_users=True,
            query_column="user_idx",
            item_column="item_idx",
        )
        train, val = train_spl.split(log)
        train = train.drop("timestamp")
        val = val.drop("timestamp")

        train = train.toPandas()
        val = val.toPandas()
        pd_item_features = item_features.toPandas()
        pd_user_features = user_features.toPandas()
        if self.cnt_users is None:
            self.cnt_users = pd_user_features.shape[0]
        if self.cnt_items is None:
            self.cnt_items = pd_item_features.shape[0]
        self.num_of_train_labels = self.cnt_items

        self.preprocess_features_fit(train, pd_item_features, pd_user_features)
        transform_user_features, transform_item_features = self.preprocess_features_transform(
            pd_item_features, pd_user_features
        )

        list_items = pd_item_features["item_idx"].values.tolist()

        dataloader_train_users = []
        train = train.set_axis(range(train.shape[0]), axis="index")
        train_group_by_users = train.groupby("user_idx")
        for idx, df_train_idx in tqdm(train_group_by_users):
            df_train_idx = df_train_idx.loc[df_train_idx["relevance"] == 1]
            if df_train_idx.shape[0] == 0:
                continue
            df_train_idx = df_train_idx.set_axis(range(df_train_idx.shape[0]), axis="index")
            train_dataset, train_dataloader = self._data_loader(
                idx, df_train_idx, transform_user_features, transform_item_features, list_items, train=True
            )
            dataloader_train_users.append(train_dataloader)

        dataloader_val_users = []
        self.dict_true_items_val = {}
        transform_item_features.sort_values(by=["item_idx"], inplace=True, ignore_index=True)
        val = val.set_axis(range(val.shape[0]), axis="index")
        val_group_by_users = val.groupby("user_idx")
        for idx, df_val_idx in tqdm(val_group_by_users):
            self.dict_true_items_val[idx] = df_val_idx.loc[(df_val_idx["relevance"] == 1)]["item_idx"].values.tolist()
            df_val = cartesian_product(pd.DataFrame({"user_idx": [idx]}), transform_item_features[["item_idx"]])
            df_val = df_val.set_axis(range(df_val.shape[0]), axis="index")
            dataloader_val_users.append(
                self._data_loader(
                    idx, df_val, transform_user_features, transform_item_features, list_items, train=False
                )
            )

        self.size_wide_features, self.size_continuous_features, self.size_cat_features = (
            train_dataset.get_size_features()
        )
        self.model = WideDeep(
            dim_head=self.dim_head,
            deep_out_dim=self.deep_out_dim,
            hidden_layers=self.hidden_layers,
            size_wide_features=self.size_wide_features,
            size_continuous_features=self.size_continuous_features,
            size_cat_features=self.size_cat_features,
            wide_out_dim=self.wide_out_dim,
            head_dropout=self.head_dropout,
            deep_dropout=self.deep_dropout,
            cnt_users=self.cnt_users,
            cnt_items=self.cnt_items,
            user_embed=self.embedding_sizes[0],
            item_embed=self.embedding_sizes[1],
            crossed_embed=self.embedding_sizes[2],
        )
        if self.use_warp_loss:
            self.criterion = warp_loss
        else:
            self.criterion = w_log_loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.opt_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, self.lr_min)

        self.train(self.model, dataloader_train_users, dataloader_val_users)

    def train(self, model, train_dataloader, val_dataloader):
        """
        Run training loop.
        """
        train_losses = []
        val_ndcg = []
        model = model.to(self.device)
        for epoch in range(self.n_epochs):
            train_loss = self._batch_pass(model, train_dataloader)
            ndcg = self.predict_val_with_ndcg(model, val_dataloader, k=10)
            train_losses.append(train_loss)
            val_ndcg.append(ndcg)

            if self.plot_dir is not None and epoch > 0:
                clear_output(wait=True)
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
                ax1.plot(train_losses, label="train", color="b")
                ax2.plot(val_ndcg, label="val_ndcg", color="g")
                size = max(1, round(epoch / 10))
                plt.xticks(range(epoch - 1)[::size])
                ax1.set_ylabel("loss")
                ax1.set_xlabel("epoch")
                ax2.set_ylabel("ndcg")
                ax2.set_xlabel("epoch")
                plt.legend()
                plt.savefig(self.plot_dir)
                plt.show()
                self.logger.info("ndcg val =%.4f", ndcg)

    def _loss(self, preds, labels):
        if self.use_warp_loss:
            ind_pos = torch.where(labels == 1)[0]
            ind_neg = torch.where(labels == 0)[0]
            min_batch = ind_pos.shape[0]
            if ind_pos.shape[0] == 0 or ind_neg.shape[0] == 0:
                return
            indexes_pos = ind_pos
            pos = preds.squeeze()[indexes_pos].unsqueeze(-1)
            list_neg = []
            for _ in range(min_batch):
                indexes_neg = ind_neg[torch.randperm(ind_neg.shape[0])]
                list_neg.append(preds.squeeze()[indexes_neg].unsqueeze(-1))
            neg = torch.cat(list_neg, dim=-1)
            neg = neg.transpose(0, 1)
            loss = self.criterion(pos, neg, self.num_of_train_labels, self.device)
        else:
            loss = self.criterion(preds.squeeze(), labels, self.device)
        return loss

    def _batch_pass(self, model, train_dataloader):
        """
        Run training one epoch loop.
        """
        model.train()
        idx = 0
        cumulative_loss = 0
        preds = None
        for user_dataloader in tqdm(train_dataloader):
            for batch in user_dataloader:
                wide_part, continuous_part, cat_part, users, items, labels = batch
                self.optimizer.zero_grad()
                preds = model(wide_part, continuous_part, cat_part, users, items)
                loss = self._loss(preds, labels)
                if loss is not None:
                    loss.backward()
                    self.optimizer.step()
                    cumulative_loss += loss.item()
                    idx += 1

        self.lr_scheduler.step()
        return cumulative_loss / idx

    def predict_val_with_ndcg(self, model, val_dataloader, k):
        """
        This function returns the NDCG metric for the validation data.
        """
        if len(val_dataloader) == 0:
            return 0

        ndcg = 0
        idx = 0
        model = model.to(self.device)
        for user_dataloader in tqdm(val_dataloader):
            _, _, _, users, _, _ = next(iter(user_dataloader))
            user = int(users[0])
            sample_pred = np.array(self.predict_val(model, user_dataloader))
            top_k_predicts = (-sample_pred).argsort()[:k]
            ndcg += (np.isin(top_k_predicts, self.dict_true_items_val[user]).sum()) / k
            idx += 1

        metric = ndcg / idx
        return metric

    def predict_val(self, model, val_dataloader):
        """
        This function returns the relevances for the validation data.
        """
        probs = []
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for wide_part, continuous_part, cat_part, users, items, _ in val_dataloader:
                preds = model(wide_part, continuous_part, cat_part, users, items)
                probs += (preds.squeeze()).tolist()
        return probs

    def predict_test(self, model, test_dataloader, cnt_samples_for_predict):
        """
        This function returns a list of cnt_samples_for_predict relevancies for each pair (users, items)
        in val_dataloader
        """
        probs = []
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for wide_part, continuous_part, cat_part, users, items, _ in test_dataloader:
                preds = model.forward_for_predict(wide_part, continuous_part, cat_part, users, items)
                probs.extend(model.forward_dropout(preds).squeeze().tolist() for __ in range(cnt_samples_for_predict))
        return probs

    def _predict(
        self,
        log: DataFrame,  # noqa: ARG002
        k: int,  # noqa: ARG002
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,  # noqa: ARG002
    ) -> DataFrame:
        if user_features is None:
            msg = "User features are missing for predict"
            raise ValueError(msg)
        if item_features is None:
            msg = "Item features are missing for predict"
            raise ValueError(msg)

        pd_users = users.toPandas()
        pd_items = items.toPandas()
        pd_user_features = user_features.toPandas()
        pd_item_features = item_features.toPandas()

        list_items = pd_item_features["item_idx"].values.tolist()

        transform_user_features, transform_item_features = self.preprocess_features_transform(
            pd_item_features, pd_user_features
        )

        preds = []
        users_ans = []
        items_ans = []
        for idx in tqdm(pd_users["user_idx"].unique()):
            df_test_idx = cartesian_product(pd.DataFrame({"user_idx": [idx]}), pd_items)
            df_test_idx = df_test_idx.set_axis(range(df_test_idx.shape[0]), axis="index")
            test_dataloader = self._data_loader(
                idx, df_test_idx, transform_user_features, transform_item_features, list_items, train=False
            )

            samples = np.array(self.predict_test(self.model, test_dataloader, self.cnt_samples_for_predict))
            sample_pred = np.mean(samples, axis=0) + self.exploration_coef * np.sqrt(np.var(samples, axis=0))

            preds += sample_pred.tolist()
            users_ans += [idx] * df_test_idx.shape[0]
            items_ans += df_test_idx["item_idx"].values.tolist()

        res_df = pd.DataFrame({"user_idx": users_ans, "item_idx": items_ans, "relevance": preds})
        pred = convert2spark(res_df)
        return pred

    @property
    def _init_args(self):
        return {
            "n_epochs": self.n_epochs,
            "union_cols": self.union_cols,
            "cnt_users": self.cnt_users,
            "cnt_items": self.cnt_items,
            "size_wide_features": self.size_wide_features,
            "size_continuous_features": self.size_continuous_features,
            "size_cat_features": self.size_cat_features,
        }

    def model_save(self, dir_name):
        """
        This function saves the model.
        """
        os.makedirs(dir_name, exist_ok=True)

        joblib.dump(self.scaler_user, os.path.join(dir_name, "scaler_user.joblib"))
        joblib.dump(self.encoder_intersept_user, os.path.join(dir_name, "encoder_intersept_user.joblib"))
        joblib.dump(self.encoder_diff_user, os.path.join(dir_name, "encoder_diff_user.joblib"))

        joblib.dump(self.scaler_item, os.path.join(dir_name, "scaler_item.joblib"))
        joblib.dump(self.encoder_intersept_item, os.path.join(dir_name, "encoder_intersept_item.joblib"))
        joblib.dump(self.encoder_diff_item, os.path.join(dir_name, "encoder_diff_item.joblib"))

        torch.save(self.model.state_dict(), os.path.join(dir_name, "model_weights.pth"))
        torch.save(
            {
                "fit_users": self.fit_users.toPandas(),
                "fit_items": self.fit_items.toPandas(),
            },
            os.path.join(dir_name, "fit_info.pth"),
        )

    def model_load(self, dir_name):
        """
        This function loads the model.
        """
        self.scaler_user = joblib.load(os.path.join(dir_name, "scaler_user.joblib"))
        self.encoder_intersept_user = joblib.load(os.path.join(dir_name, "encoder_intersept_user.joblib"))
        self.encoder_diff_user = joblib.load(os.path.join(dir_name, "encoder_diff_user.joblib"))

        self.scaler_item = joblib.load(os.path.join(dir_name, "scaler_item.joblib"))
        self.encoder_intersept_item = joblib.load(os.path.join(dir_name, "encoder_intersept_item.joblib"))
        self.encoder_diff_item = joblib.load(os.path.join(dir_name, "encoder_diff_item.joblib"))

        self.model = WideDeep(
            dim_head=self.dim_head,
            deep_out_dim=self.deep_out_dim,
            hidden_layers=self.hidden_layers,
            size_wide_features=self.size_wide_features,
            size_continuous_features=self.size_continuous_features,
            size_cat_features=self.size_cat_features,
            wide_out_dim=self.wide_out_dim,
            head_dropout=self.head_dropout,
            deep_dropout=self.deep_dropout,
            cnt_users=self.cnt_users,
            cnt_items=self.cnt_items,
            user_embed=self.embedding_sizes[0],
            item_embed=self.embedding_sizes[1],
            crossed_embed=self.embedding_sizes[2],
        )
        self.model.load_state_dict(torch.load(os.path.join(dir_name, "model_weights.pth")))

        checkpoint = torch.load(os.path.join(dir_name, "fit_info.pth"))
        self.fit_users = convert2spark(checkpoint["fit_users"])
        self.fit_items = convert2spark(checkpoint["fit_items"])
