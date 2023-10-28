import math
import numpy as np
import pandas as pd
import os

from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.metrics import Metric, NDCG
from replay.models.base_rec import HybridRecommender
from replay.models import LinUCB
from replay.utils import convert2spark

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import random_split
import torch.utils.data as td
from tqdm import tqdm_notebook, tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from collections import Counter
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
import copy
from replay.splitters import DateSplitter

pd.options.mode.chained_assignment = None

#export
import torch

def num_tries_gt_zero(scores, batch_size, max_trials, max_num, device):
    '''
    scores: [batch_size x N] float scores
    returns: [batch_size x 1] the lowest indice per row where scores were first greater than 0. plus 1
    '''
    tmp = scores.gt(0).nonzero().t()
    # We offset these values by 1 to look for unset values (zeros) later
    values = tmp[1] + 1
    # TODO just allocate normal zero-tensor and fill it?
    # Sparse tensors can't be moved with .to() or .cuda() if you want to send in cuda variables first
    if device.type == 'cuda':
        t = torch.cuda.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials+1))).to_dense()
    else:
        t = torch.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials+1))).to_dense()
    t[(t == 0)] += max_num # set all unused indices to be max possible number so its not picked by min() call

    tries = torch.min(t, dim=1)[0]
    return tries


def warp_loss(positive_predictions, negative_predictions, num_labels, device):
    '''
    positive_predictions: [batch_size x 1] floats between -1 to 1
    negative_predictions: [batch_size x N] floats between -1 to 1
    num_labels: int total number of labels in dataset (not just the subset you're using for the batch)
    device: pytorch.device
    '''
    batch_size, max_trials = negative_predictions.size(0), negative_predictions.size(1)

    offsets, ones, max_num = (torch.arange(0, batch_size, 1).long().to(device) * (max_trials + 1),
                              torch.ones(batch_size, 1).float().to(device),
                              batch_size * (max_trials + 1) )

    sample_scores = (1 + negative_predictions - positive_predictions)
    # Add column of ones so we know when we used all our attempts, This is used for indexing and computing should_count_loss if no real value is above 0
    sample_scores, negative_predictions = (torch.cat([sample_scores, ones], dim=1),
                                           torch.cat([negative_predictions, ones], dim=1))

    tries = num_tries_gt_zero(sample_scores, batch_size, max_trials, max_num, device)
    attempts, trial_offset = tries.float(), (tries - 1) + offsets
    loss_weights, should_count_loss = ( torch.log(torch.floor((num_labels - 1) / attempts)),
                                        (attempts <= max_trials).float()) #Don't count loss if we used max number of attempts

    losses = loss_weights * ((1 - positive_predictions.view(-1)) + negative_predictions.view(-1)[trial_offset]) * should_count_loss

    return losses.sum()

def cartesian_product_basic(left, right):
    return (left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))
    
class SamplerWithReset(td.SequentialSampler):
    def __iter__(self):
        self.data_source.reset()
        return super().__iter__()
        
class MyDatasetreset(torch.utils.data.Dataset):
    def __init__(self, idx, log_train, user_features, item_features, list_items, union_cols, cnt_neg_samples, device, target: str = None):
        if cnt_neg_samples is not None:
            self.cnt_neg_samples = cnt_neg_samples
            self.user_features = user_features
            self.item_features = item_features
            item_idx_user = log_train['item_idx'].values.tolist()
            self.item_idx_not_user = list(set(list_items).difference(set(item_idx_user)))
        else:
            self.cnt_neg_samples = cnt_neg_samples
            self.user_features = None
            self.item_features = None
            self.item_idx_not_user = None
        self.device = device
        self.union_cols = union_cols
        dataframe = log_train.merge(user_features, on='user_idx', how='inner')
        self.dataframe = dataframe.merge(item_features, on='item_idx', how='inner')
        self.user_idx = idx
        self.data_sample = None
        self.wide_part = Tensor(self.dataframe[self.union_cols["wide_cols"]].to_numpy().astype('float32')).to(self.device)
        self.continuous_part = Tensor(self.dataframe[self.union_cols["continuous_cols"]].to_numpy().astype('float32')).to(self.device)
        self.cat_part = Tensor(self.dataframe[self.union_cols["cat_embed_cols"]].to_numpy().astype('float32')).to(self.device)
        self.users = Tensor(self.dataframe[['user_idx']].to_numpy().astype('int')).to(torch.long).to(self.device)
        self.items = Tensor(self.dataframe[['item_idx']].to_numpy().astype('int')).to(torch.long).to(self.device)
        if target is not None:
            self.target = Tensor(dataframe[target].to_numpy().astype('int')).to(self.device)
        else:
            self.target = target
        self.target_column = target
            
    def get_parts(self, data_sample):
        self.wide_part = Tensor(data_sample[self.union_cols["wide_cols"]].to_numpy().astype('float32')).to(self.device)
        self.continuous_part = Tensor(data_sample[self.union_cols["continuous_cols"]].to_numpy().astype('float32')).to(self.device)
        self.cat_part = Tensor(data_sample[self.union_cols["cat_embed_cols"]].to_numpy().astype('float32')).to(self.device)
        self.users = Tensor(data_sample[['user_idx']].to_numpy().astype('int')).to(torch.long).to(self.device)
        self.items = Tensor(data_sample[['item_idx']].to_numpy().astype('int')).to(torch.long).to(self.device)
        if self.target_column is not None:
            self.target = Tensor(data_sample[self.target_column].to_numpy().astype('int')).to(self.device)
        else:
            self.target = self.target_column

    def __getitem__(self, idx):
        
        target = -1
        if self.target is not None:
            target = self.target[idx]
        return self.wide_part[idx], self.continuous_part[idx], self.cat_part[idx], self.users[idx], self.items[idx], target

    def __len__(self):
        if self.data_sample is not None:
            return self.data_sample.shape[0]
        else:
            return self.dataframe.shape[0]
    
    def get_size_features(self):
        return self.wide_part.shape[1], self.continuous_part.shape[1], self.cat_part.shape[1]
    
    def reset(self):
        n_samples = min(len(self.item_idx_not_user), self.cnt_neg_samples)
        if n_samples > 0:
            sample_item = np.random.choice(self.item_idx_not_user, n_samples, replace=False)
            sample_item_feat = self.item_features.loc[self.item_features["item_idx"].isin(sample_item)]
            sample_item_feat = sample_item_feat.set_axis(range(sample_item_feat.shape[0]), axis='index')
            df_sample = cartesian_product_basic(self.user_features.loc[self.user_features['user_idx'] == self.user_idx], sample_item_feat)
            df_sample[self.target_column]=0
            self.data_sample = pd.concat([self.dataframe, df_sample], axis=0, ignore_index=True)
            self.get_parts(self.data_sample)
        return

class Wide(nn.Module):
    def __init__(self, 
        input_dim: int, 
        out_dim: int = 1
        ):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)
            )
        self.out_dim = out_dim
        
    def forward(self, input):
        output = self.linear(input)
        return output

class Deep(nn.Module):
    def __init__(self, 
        input_dim: int,
        out_dim: int,
        hidden_layers: List[int],
        deep_dropout: float
        ):
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
        
    def forward(self, input):
        output = self.deep_model(input)
        return output
        
class Embed_model(nn.Module):
    def __init__(self, 
        cnt_users: int,
        cnt_items: int,
        user_embed: int,
        item_embed: int,
        crossed_embed: int
        ):
        super().__init__()
        self.user_embed = nn.Embedding(num_embeddings=cnt_users, embedding_dim=user_embed)
        self.item_embed = nn.Embedding(num_embeddings=cnt_items, embedding_dim=item_embed)
        self.user_crossed_embed = nn.Embedding(num_embeddings=cnt_users, embedding_dim=crossed_embed)
        self.item_crossed_embed = nn.Embedding(num_embeddings=cnt_items, embedding_dim=crossed_embed)
    def forward(self, users, items):
        users_to_embed = self.user_embed(users).squeeze()
        items_to_embed = self.item_embed(items).squeeze()
        cross_users = self.user_crossed_embed(users).squeeze()
        cross_items = self.item_crossed_embed(items).squeeze()
        cross = (cross_users*cross_items).sum(dim=-1).unsqueeze(-1)
        return users_to_embed, items_to_embed, cross_users, cross_items, cross
        
class WideDeep(nn.Module):
    def __init__(self,
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
        crossed_embed: int
        ):
        super().__init__()
        # self.embedding_layer = nn.Sequential(
        #     nn.Linear(size_cat_features, embedding_size),
        #     nn.BatchNorm1d(embedding_size),
        # )
        self.embed_model = Embed_model(
            cnt_users,
            cnt_items,
            user_embed,
            item_embed,
            crossed_embed
            ) 
        self.Wide = Wide(size_wide_features+crossed_embed*2+1, wide_out_dim)
        self.Deep = Deep(size_cat_features+size_continuous_features + user_embed + item_embed, deep_out_dim, hidden_layers, deep_dropout)
        self.head_model = nn.Sequential(
            nn.Linear(wide_out_dim + deep_out_dim, dim_head),
            nn.ReLU()
            )
        self.last_layer = nn.Sequential(
            nn.Linear(dim_head, 1)
            )
        self.head_dropout = head_dropout 
    
    def my_forward(self, wide_part, continuous_part, cat_part, users_to_embed, items_to_embed, cross_users, cross_items, cross):
        input_deep = torch.cat((cat_part, continuous_part, users_to_embed, items_to_embed), dim=-1).squeeze()
        out_deep = self.Deep(input_deep)
        wide_part = torch.cat((wide_part, cross_users, cross_items, cross), dim=-1)
        out_wide = self.Wide(wide_part)
        input = torch.cat((out_wide, out_deep), dim=-1)
        out = self.head_model(input)
        out = nn.functional.dropout(out, p=self.head_dropout, training=True)
        out = self.last_layer(out)
        return out
        
    def forward(self, wide_part, continuous_part, cat_part, users, items):
        users_to_embed, items_to_embed, cross_users, cross_items, cross = self.embed_model(users, items)
        out = self.my_forward(wide_part, continuous_part, cat_part, users_to_embed, items_to_embed, cross_users, cross_items, cross)
        return out
    
def w_log_loss(input, target, device):
    input = torch.nn.functional.sigmoid(input)
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    count_1 = target.sum().item()
    count_0 = target.shape[0] - count_1
    class_count=np.array([count_0,count_1])
    if count_1 == 0 or count_0 == 0:
        weight= np.array([1.0, 1.0])
    else:
        weight= np.max(class_count)/class_count
    weight = Tensor(weight).to(device)
    loss = weight[1] * target * torch.log(input) + weight[0] * (1 - target)*torch.log(1-input)
    return -loss.mean()
        
class NeuralTSModified(HybridRecommender):
    def __init__(
        self,
        user_cols: Dict[str,List[str]] = None,
        item_cols: Dict[str,List[str]] = None,
        dim_head: int = 4,
        deep_out_dim: int = 8,
        hidden_layers: List[int] = [128,64,32],
        embedding_sizes: List[int] = [32, 8 ,64],
        wide_out_dim: int = 1,
        head_dropout: float = 0.0,
        deep_dropout: float = 0.0,
        n_epochs: int = 10,
        lr: float = 3e-4,
        lr_min: float = 1e-5,
        use_gpu: bool = False,
        plot: bool = False,
        is_warp_loss: bool = False,
        cnt_neg_samples: int = 100,
        cnt_samples_for_predict: int = 10,
        eps: float = 0.0
        
        
    ):
        self.user_cols = user_cols
        self.item_cols = item_cols
        self.dim_head = dim_head
        self.deep_out_dim = deep_out_dim
        self.hidden_layers = hidden_layers
        self.wide_out_dim = wide_out_dim
        self.head_dropout = head_dropout
        self.deep_dropout = deep_dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_min = lr_min
        self.device = torch.device("cpu")
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plot = plot
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.is_warp_loss = is_warp_loss
        self.cnt_neg_samples = cnt_neg_samples
        self.embedding_sizes = embedding_sizes
        self.cnt_samples_for_predict = cnt_samples_for_predict
        self.eps = eps
        
    def train_model(self, model, train_dataloader, val_dataloader, device, n_epochs, plot=False, is_warp_loss = False):
        train_loss = []
        val_loss = []
        val_ndcg = []
        model = model.to(device)
        for epoch in range(n_epochs):           
            loss = self.train_one_epoch_batch_user(model, train_dataloader, device, is_warp_loss)
            ndcg = self.predict_val_with_ndcg(model, val_dataloader, device, k=10)
            val_ndcg.append(ndcg)
            train_loss.append(loss)
            if plot and epoch > 0:
                clear_output(wait = True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
                ax1.plot(train_loss, label = 'train', color = 'b')
                ax2.plot(val_ndcg, label = 'val_ndcg', color = 'g')
                sz = max(1, round(epoch/10))
                plt.xticks(range(epoch -1)[::sz])
                ax1.set_ylabel('loss')
                ax1.set_xlabel('epoch')
                ax2.set_ylabel('ndcg')
                ax2.set_xlabel('epoch')
                plt.legend()
                plt.show()
                print("ndcg val =", ndcg)
        return
    
    def train_one_epoch_batch_user(self, model, train_dataloader, device, is_warp_loss):
        model.train()
        idx = 0
        cumulative_loss = 0
        preds=None
        for user_dataloader in tqdm(train_dataloader):
            for (wide_part, continuous_part, cat_part, users, items, labels) in user_dataloader:
                wide_part = wide_part
                continuous_part = continuous_part
                cat_part = cat_part
                users = users
                items = items
                labels = labels
                self.optimizer.zero_grad()
                preds = model(wide_part, continuous_part, cat_part, users, items)
                if is_warp_loss:
                    ind_pos = torch.where(labels == 1)[0]
                    ind_neg = torch.where(labels == 0)[0]
                    min_batch = ind_pos.shape[0]
                    if ind_pos.shape[0] == 0 or ind_neg.shape[0] == 0:
                        continue 
                    indexes_pos = ind_pos
                    pos = preds.squeeze()[indexes_pos].unsqueeze(-1)
                    list_neg = []
                    for i in range(min_batch):
                        indexes_neg = ind_neg[torch.randperm(ind_neg.shape[0])]
                        list_neg.append(preds.squeeze()[indexes_neg].unsqueeze(-1))
                    neg = torch.cat(list_neg, dim = -1)
                    neg = neg.transpose(0,1)
                    loss = self.criterion(pos, neg, self.num_of_train_labels, device)
                else:
                    loss = self.criterion(preds.squeeze(), labels, device)
                loss.backward()
                self.optimizer.step()
                cumulative_loss += loss.item()
                idx += 1
        self.lr_scheduler.step()
        return cumulative_loss / idx,
    
    def predict_val_with_ndcg(self, model, val_dataloader, device, k):
        ndcg = 0
        idx = 0
        for user_dataloader in tqdm(val_dataloader):
            _, _, _, users, _, _ = next(iter(user_dataloader))
            user = int(users[0])
            sample_pred = np.array(self.predict_test(self.model, user_dataloader, self.device))
            top_k_predicts = (-sample_pred).argsort()[:k]
            ndcg += (np.isin(top_k_predicts, self.dict_true_items_val[user]).sum())/k
            idx += 1
        return ndcg/idx            
    
    def predict_test(self, model, val_dataloader, device):
        probs = []
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for idx, (wide_part, continuous_part, cat_part, users, items, labels) in enumerate(val_dataloader): 
                wide_part = wide_part
                continuous_part = continuous_part
                cat_part = cat_part
                users = users
                items = items
                preds = model(wide_part, continuous_part, cat_part, users, items)
                probs += (preds.squeeze()).tolist()
        return probs
            
    def preproces_features_fit(self, train, item_features, user_features):
        train_users = user_features.loc[user_features['user_idx'].isin(train['user_idx'].values.tolist())]
        wide_cols_cat = list(set(self.user_cols['cat_embed_cols']) & set(self.user_cols['wide_cols']))
        cat_embed_cols_not_wide = list(set(self.user_cols['cat_embed_cols']).difference(set(wide_cols_cat)))
        if len(self.user_cols['continuous_cols']) != 0:
            self.scaler_user = MinMaxScaler()
            self.scaler_user.fit(train_users[self.user_cols['continuous_cols']])
        if len(wide_cols_cat) != 0:
            self.myEncoder_intersept_user = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.myEncoder_intersept_user.fit(train_users[wide_cols_cat])
        if len(cat_embed_cols_not_wide)!=0:
            self.myEncoder_diff_user = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.myEncoder_diff_user.fit(train_users[cat_embed_cols_not_wide])
        train_items = item_features.loc[item_features['item_idx'].isin(train['item_idx'].values.tolist())]
        wide_cols_cat = list(set(self.item_cols['cat_embed_cols']) & set(self.item_cols['wide_cols']))
        cat_embed_cols_not_wide = list(set(self.item_cols['cat_embed_cols']).difference(set(wide_cols_cat)))
        if len(self.item_cols['continuous_cols']) != 0:
            self.scaler_item = MinMaxScaler()
            self.scaler_item.fit(train_items[self.item_cols['continuous_cols']])
        if len(wide_cols_cat) != 0:
            self.myEncoder_intersept_item = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.myEncoder_intersept_item.fit(train_items[wide_cols_cat])
        if len(cat_embed_cols_not_wide)!=0:
            self.myEncoder_diff_item = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.myEncoder_diff_item.fit(train_items[cat_embed_cols_not_wide])
        return
        
    def preproces_features_trasform(self, item_features, user_features):
        self.union_cols = {'continuous_cols':[], 'cat_embed_cols':[], 'wide_cols':[]}
        wide_cols_cat = list(set(self.user_cols['cat_embed_cols']) & set(self.user_cols['wide_cols']))
        cat_embed_cols_not_wide = list(set(self.user_cols['cat_embed_cols']).difference(set(wide_cols_cat)))
        if len(self.user_cols['continuous_cols']) != 0:
            users_continuous = pd.DataFrame(self.scaler_user.transform(user_features[self.user_cols['continuous_cols']]), columns= self.user_cols['continuous_cols'])
            self.union_cols['continuous_cols'] += self.user_cols['continuous_cols']
        else:
            users_continuous = user_features[[]]
        if len(wide_cols_cat) != 0:
            users_wide_cat = pd.DataFrame(self.myEncoder_intersept_user.transform(user_features[wide_cols_cat]),
                                            columns=list(self.myEncoder_intersept_user.get_feature_names_out(wide_cols_cat)))
            self.union_cols['cat_embed_cols'] += list(self.myEncoder_intersept_user.get_feature_names_out(wide_cols_cat))
            self.union_cols['wide_cols'] += list(set(self.user_cols['wide_cols']).difference(set(self.user_cols['cat_embed_cols']))) + list(self.myEncoder_intersept_user.get_feature_names_out(wide_cols_cat))  
        else:
            users_wide_cat = user_features[[]]
        if len(cat_embed_cols_not_wide)!=0:
            users_cat = pd.DataFrame(self.myEncoder_diff_user.transform(user_features[cat_embed_cols_not_wide]),
                                            columns=list(self.myEncoder_diff_user.get_feature_names_out(cat_embed_cols_not_wide)))
            self.union_cols['cat_embed_cols'] += list(self.myEncoder_diff_user.get_feature_names_out(cat_embed_cols_not_wide))
        else:
            users_cat = user_features[[]]
        transform_user_features = pd.concat(
            [user_features[['user_idx']], users_continuous, users_wide_cat, users_cat], 
            axis=1
            )
        wide_cols_cat = list(set(self.item_cols['cat_embed_cols']) & set(self.item_cols['wide_cols']))
        cat_embed_cols_not_wide = list(set(self.item_cols['cat_embed_cols']).difference(set(wide_cols_cat)))
        if len(self.item_cols['continuous_cols']) != 0:
            items_continuous = pd.DataFrame(self.scaler_item.transform(item_features[self.item_cols['continuous_cols']]), columns= self.item_cols['continuous_cols'])
            self.union_cols['continuous_cols'] += self.item_cols['continuous_cols']
        else:
            items_continuous = item_features[[]]
        if len(wide_cols_cat) != 0:
            items_wide_cat = pd.DataFrame(self.myEncoder_intersept_item.transform(item_features[wide_cols_cat]),
                                            columns=list(self.myEncoder_intersept_item.get_feature_names_out(wide_cols_cat)))
            self.union_cols['cat_embed_cols'] += list(self.myEncoder_intersept_item.get_feature_names_out(wide_cols_cat))
            self.union_cols['wide_cols'] += list(set(self.item_cols['wide_cols']).difference(set(self.item_cols['cat_embed_cols']))) + list(self.myEncoder_intersept_item.get_feature_names_out(wide_cols_cat))
        else:
            items_wide_cat = item_features[[]]
        if len(cat_embed_cols_not_wide)!=0:
            items_cat = pd.DataFrame(self.myEncoder_diff_item.transform(item_features[cat_embed_cols_not_wide]),
                                            columns=list(self.myEncoder_diff_item.get_feature_names_out(cat_embed_cols_not_wide)))
            self.union_cols['cat_embed_cols'] += list(self.myEncoder_diff_item.get_feature_names_out(cat_embed_cols_not_wide))
        else: 
            items_cat = item_features[[]]
        transform_item_features = pd.concat(
            [item_features[['item_idx']], items_continuous, items_wide_cat, items_cat], 
            axis=1
            )
        return transform_user_features, transform_item_features        

    @property
    def _init_args(self):
        return {
            "n_epochs": self.n_epochs,
            "plot": self.plot
        }
    
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        #should not work if user features or item features are unavailable 
        if user_features is None:
            raise ValueError("User features are missing for fitting")
        if item_features is None:
            raise ValueError("Item features are missing for fitting")
        #assuming that user_features and item_features are both dataframes
        #common dataframe        
        train_spl = DateSplitter(
                        test_start=0.2,
                        drop_cold_items=True,
                        drop_cold_users=True,
                    )
        train, val = train_spl.split(log)
        train = train.drop('timestamp')
        val = val.drop('timestamp')
        
        train = train.toPandas()
        val = val.toPandas()
        pd_item_features = item_features.toPandas()
        pd_user_features = user_features.toPandas()
        self.cnt_users = pd_user_features.shape[0]
        self.cnt_items = pd_item_features.shape[0]
        list_items = item_features.toPandas()['item_idx'].values.tolist()
        self.preproces_features_fit(train, pd_item_features, pd_user_features)
        transform_user_features, transform_item_features = self.preproces_features_trasform(pd_item_features, pd_user_features)
   
        target = 'relevance'
        self.num_of_train_labels = self.cnt_items
        train = train.set_axis(range(train.shape[0]), axis = 'index')
        val = val.set_axis(range(val.shape[0]), axis = 'index')
        
    
        #for batch by users
        dataloader_train_users = []
        size_wide_features, size_continuous_features, size_cat_features = 0, 0, 0
        train_group_by_users = train.groupby("user_idx")
        for idx, df_train_idx in tqdm(train_group_by_users):  
            df_train_idx = df_train_idx.loc[df_train_idx[target]==1]
            if df_train_idx.shape[0] == 0:
                continue
            df_train_idx = df_train_idx.set_axis(range(df_train_idx.shape[0]), axis='index')
            dataset = MyDatasetreset(idx, df_train_idx, transform_user_features, transform_item_features, list_items, self.union_cols, self.cnt_neg_samples, self.device, target)
            sampler = SamplerWithReset(dataset)
            dataloader_train_users.append(torch.utils.data.DataLoader(dataset, 
                                                     batch_size=df_train_idx.shape[0]+self.cnt_neg_samples,  
                                                     sampler=sampler))
            size_wide_features, size_continuous_features, size_cat_features= dataset.get_size_features()
            
        dataloader_val_users = []
        self.dict_true_items_val = {}
        transform_item_features.sort_values(by=['item_idx'], inplace=True, ignore_index=True) 
        target = None
        val_group_by_users = val.groupby("user_idx")
        for idx, df_val_idx in tqdm(val_group_by_users):
            self.dict_true_items_val[idx] = df_val_idx.loc[(df_val_idx['relevance']==1)]['item_idx'].values.tolist()
            #sample_users = pd.concat([pd_user_features.loc[pd_user_features['user_idx'] == idx]]*self.cnt_items, axis=0, ignore_index=True)
            df_val = cartesian_product_basic(pd.DataFrame({'user_idx': [idx]}), transform_item_features[['item_idx']])
            df_val = df_val.set_axis(range(df_val.shape[0]), axis='index')
            dataset = MyDatasetreset(idx, df_val, transform_user_features, transform_item_features, list_items, self.union_cols, None, self.device, target)
            dataloader_val_users.append(torch.utils.data.DataLoader(dataset, 
                                                     batch_size=df_val.shape[0], 
                                                     shuffle=False))
        self.model = WideDeep(
                    dim_head=self.dim_head,
                    deep_out_dim=self.deep_out_dim,
                    hidden_layers=self.hidden_layers,
                    size_wide_features=size_wide_features,
                    size_continuous_features=size_continuous_features,
                    size_cat_features=size_cat_features,
                    wide_out_dim=self.wide_out_dim,
                    head_dropout=self.head_dropout,
                    deep_dropout=self.deep_dropout,
                    cnt_users=self.cnt_users,
                    cnt_items=self.cnt_items,
                    user_embed = self.embedding_sizes[0],
                    item_embed = self.embedding_sizes[1],
                    crossed_embed = self.embedding_sizes[2],
                )
        if self.is_warp_loss:
            self.criterion = warp_loss
        else:
            self.criterion = w_log_loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, self.lr_min)
        self.train_model(self.model, dataloader_train_users, dataloader_val_users, self.device, self.n_epochs, self.plot, self.is_warp_loss)
        return 
        
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        if user_features is None or item_features is None:
            raise ValueError("Can not make predict in the Neural TS method")
                        
        #preprocess
        # log_pred = users.crossJoin(items.select("item_idx")).select("user_idx", "item_idx")
        # log_pred = log_pred.toPandas()
        # val_group_by_users = log_pred.groupby("user_idx")
        pd_users = users.toPandas()
        pd_items = items.toPandas()
     
        pd_user_features = user_features.toPandas()
        pd_item_features = item_features.toPandas()
        list_items = pd_item_features['item_idx'].values.tolist()
        transform_user_features, transform_item_features = self.preproces_features_trasform(pd_item_features, pd_user_features)
        target=None
        preds = []
        users_ans = []
        items_ans = []
        #preprocess
        for idx in tqdm(pd_users['user_idx'].unique()):
            df_test_idx = cartesian_product_basic(pd.DataFrame({'user_idx': [idx]}), pd_items)
            df_test_idx = df_test_idx.set_axis(range(df_test_idx.shape[0]), axis = 'index')
            users_ans += [idx] * df_test_idx.shape[0]
            items_ans += df_test_idx['item_idx'].values.tolist()
            dataset = MyDatasetreset(idx, df_test_idx, transform_user_features, transform_item_features, list_items, self.union_cols, None, self.device, target)
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                     batch_size=df_test_idx.shape[0],  
                                                     shuffle=False)
            #predict
            samples = []
            for i in range(self.cnt_samples_for_predict):
                samples.append(self.predict_test(self.model, dataloader, self.device))
            samples = np.array(samples)
            mean = np.mean(samples, axis = 0)
            var = ((samples - mean)**2).mean(axis = 0)
            sample_pred = mean + (self.eps)*np.sqrt(var)
            preds += sample_pred.tolist()
        #return everything in a PySpark template
        res_df = pd.DataFrame(
        {'user_idx': users_ans, 'item_idx': items_ans, 'relevance': preds}
        )
        pred = convert2spark(res_df)
        return pred
