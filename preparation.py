"""
This file works on preparing datasets into input shape of Graph Neural Network.
Specifically, It is for the preparation of representation learning.
The parsing part is going to be updated for customized dataset.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings(action = "ignore")
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul
from sklearn.model_selection import train_test_split
# load the dataset
charge = pd.read_csv("charge.csv")
charge_add = pd.read_csv("charge_before.csv")
patron = pd.read_csv("patron_erica.csv")
patron_add = pd.read_csv("patron_erica_before.csv")
item = pd.read_csv("item_erica.csv")
item_add = pd.read_csv("item_erica_before.csv")
charge = pd.concat([charge, charge_add], axis = 0)
patron = pd.concat([patron, patron_add], axis = 0)
item = pd.concat([item, item_add], axis = 0)
charge.drop_duplicates(inplace=True)
patron.drop_duplicates(inplace = True)
item.drop_duplicates(inplace = True)
def seed_fix(seed: int):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os 
    torch.manual_seed(seed)
seed = 0
seed_fix(0)


# Node: patron, item / Edge: charge
def refinement(patron, item, charge):
    """
    charge - ID: 대출 ID, PATRON_ID: 이용자 ID, ITEM_ID: 도서 ID
    patron - ID: 이용자 ID, PATRON_TYPE_ID: 이용자의 정보, 상위소속명, 소속명, GENDER, 생년
    item - ID: 도서 ID, TITLE: 책 제목, AUTHOR: 저자 정보, charge: 대출 횟수, hold: 예약 여부

    patron과 item을 charge를 통한 connection 진행.
    Representation Learning을 통해 적절한 도서 ID를 반환하는 것을 목표로 한다.
    point 1: patron, item의 Representation
    point 2: GNN Model building
    point 3: train the GNN model
    point 4: inference result - item's ID
    """
    assert patron is not None,\
        "이용자 정보가 존재하지 않는다."
    assert item is not None,\
        "도서 정보가 존재하지 않는다."
    assert charge is not None,\
        "대출 정보가 존재하지 않는다."
        
    patron.dropna(inplace = True)
    patron.reset_index(inplace = True)
    patron.drop(["index"], axis = 1, inplace = True)
    item.dropna(inplace = True)
    item.reset_index(inplace = True)
    item.drop(["index"], axis = 1, inplace = True)
    charge.dropna(inplace = True)
    charge.reset_index(inplace = True)
    charge.drop(["index"], axis = 1, inplace = True)
    patron.index = patron["ID"]
    item.index = item["ID"]
    patron.drop(["ID"], axis = 1, inplace = True)
    item.drop(["ID"], axis = 1, inplace = True)
    # 추후에 TITLE에 대해서는 Representation Learning을 통해 도출된 추천 TITLE ID와 Mapping을 하여 도서를 반환하는 시스템으로 간다.
    item.drop(["TITLE", "AUTHOR", "PUBLISHER", "PUBLISH_YEAR"], axis = 1, inplace = True)
    charge.drop(["ID", "CHARGE_DATE", "DISCHARGE_DATE"], axis = 1, inplace = True)
    patron.drop(["상위소속명", "PATRON_TYPE_ID"], axis = 1, inplace = True)
    patron = patron.rename(columns = {"소속명": "MAJOR"})
    return patron, item, charge

patron, item, charge = refinement(patron, item, charge)

def labelEncoding(df):
    label_encoder = LabelEncoder()
    df["MAJOR"] = label_encoder.fit_transform(df["MAJOR"])
    df["GENDER"] = label_encoder.fit_transform(df["GENDER"])
    return df

def preprocessing(user, item, charge):
    user = labelEncoding(user)
    user_mapping = {idx: i for i, idx in enumerate(user.index.unique())}
    item_mapping = {idx: i for i, idx in enumerate(item.index.unique())}
    num_user, num_item = len(user.index.unique()), len(item.index.unique())
    # charge 정보를 통해서 최대한 연결을 하고자 한다.
    is_charged = charge["ITEM_ID"].unique()
    is_charged_bool = dict()
    for idx in is_charged:
        is_charged_bool[idx] = True
    # charge 정보를 기반으로 edge index를 생성한다.
    # 유의미한 charge 정보를 연결한다.
    edge_index = [[], []]
    for idx in range(len(charge)):
        book = charge.loc[idx, "ITEM_ID"]
        users = charge.loc[idx, "PATRON_ID"]
        boolean = is_charged_bool[book]
        if boolean and book in item_mapping.keys() and users in user_mapping.keys():
            # User 정보를 고유값으로 매칭 하기 보다는 unique로 continuous하게 진행했던 것을 edge_index로 넣어버린다.
            # 마찬가지로 item 역시 고유값으로 매칭하기 보다는 continuous한 것으로 mapping한다. 추후에 Return할 때, 다시 dictionary를 이용해서 재확인한다.
            edge_index[0].append(user_mapping[users])
            edge_index[1].append(item_mapping[book])
    edge_index = torch.tensor(edge_index)
    return edge_index, num_user, num_item, \
        item_mapping, user_mapping, user, item

"""
edge_index, num_users, num_items, mapping, dataframe 생성
edge_index는 charge 정보를 기반으로 만들어졌다.
edge_index 생성 시 실제로 대출을 한 것을 기반으로 생성하였다.
"""
edge_index, num_users, num_items, item_map, user_map,\
    user, item = preprocessing(patron, item, charge)

# Dataset Preparation
nums = edge_index.size(1)
indices = np.arange(nums)
ind_train, ind_val = train_test_split(indices, test_size = 0.2, random_state = 0)
# ind_valid, ind_test = train_test_split(ind_val, test_size = 0.5, random_state = 0)

def generate_edge(edge_indices):
    '''
    edge_indices: An array representing the indices of edges in the dataset
    edge_index_sparse: SparseTensor, A sparse Tensor representing the adjacency matrix for the subset of edges.
    src, tgt가 묶여 나오게 된다.
    '''
    sub_edge_index = edge_index[:, edge_indices]
    num_nodes = num_users + num_items
    edge_index_sparse = SparseTensor(row = sub_edge_index[0],
                                     col = sub_edge_index[1],
                                     sparse_sizes = (num_nodes, num_nodes))
    return sub_edge_index, edge_index_sparse
    
train_edge_idx, train_sparse_edge_idx = generate_edge(ind_train)
valid_edge_idx, valid_sparse_edge_idx = generate_edge(ind_val)
# test_edge_idx, test_sparse_edge_idx = generate_edge(ind_test)

def batch_sample(batch_size, edge_index):
    """
    user information setting
    positive, negative sampling,
    composing batches
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim = 0) 
    indices = torch.randperm(edges.shape[1])[:batch_size]
    batch = edges[:, indices]
    user_idxs, pos_idxs, neg_idxs = batch[0], batch[1], batch[2]
    return user_idxs, pos_idxs, neg_idxs

def get_positive_items(edge_index):
    """
    parameters:
      edge_index (torch.Tensor): The edge index representing the user-item interactions.
    returns:
      pos_items (torch.Tensor): A list containing the positive items for all users.
    """
    pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in pos_items:
            pos_items[user] = []
        pos_items[user].append(item)
    return pos_items