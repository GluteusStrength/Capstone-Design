"""
Training Part
model: LightGCN
train the relation between user-item so that a model returns the interaction.
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
import preparation
from tqdm import tqdm
import numpy as np
from torch_sparse import matmul
from torch_geometric.utils import structured_negative_sampling
import warnings
warnings.filterwarnings("ignore")
from preparation import user_map, item_map, get_positive_items
from model import LightGCN
# from sklearn.metrics.pairwise import cosine_similarity


def bpr_loss(users_emb, user_emb_0, pos_emb, pos_emb_0, neg_emb, neg_emb_0, lambda_val):
    """
    Calculate the Bayesian Personalzied Ranking loss.

    parameters:
    users_emb (torch.Tensor): The final output of user embedding
    user_emb_0 (torch.Tensor): The initial user embedding
    pos_emb (torch.Tensor):  The final positive item embedding 
    pos_emb_0 (torch.Tensor): The initial item embedding
    neg_emb (torch.Tensor): The final negtive item embedding
    neg_emb_0 (torch.Tensor): The inital negtive item embedding
    lambda_val (float): L2 regulatization strength

    returns:
    loss (float): The BPR loss
    """
    pos_scores = torch.sum(users_emb * pos_emb, dim=1) 
    neg_scores = torch.sum(users_emb * neg_emb, dim=1)
    losses = -torch.log(torch.sigmoid(pos_scores - neg_scores))
    loss = torch.mean(losses) + lambda_val * \
    (torch.norm(user_emb_0) + torch.norm(pos_emb_0) + torch.norm(neg_emb_0))
    
    return loss


def evaluation(model, edge_index, sparse_edge_index, mask_index: None, k, lambda_val):
    """
    Evaluates model loss and metrics including recall, precision on the 
    Parameters:
    model: LightGCN model to evaluate.
    edge_index (torch.Tensor): Edges for the split to evaluate.
    sparse_edge_index (torch.SparseTensor): Sparse adjacency matrix.
    mask_index(torch.Tensor): Edges to remove from evaluation, in the form of a list.
    k (int): Top k items to consider for evaluation.

    Returns: loss, recall, precision
        - loss: The loss value of the model on the given split.
        - recall: The recall value of the model on the given split.
        - precision: The precision value of the model on the given split.
    """
    # get embeddings and calculate the loss
    users_emb, users_emb_0, items_emb, items_emb_0 = model.forward(edge_index)
    edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
    
    user_indices, pos_indices, neg_indices = edges[0], edges[1], edges[2]
    users_emb, users_emb_0 = users_emb[user_indices], users_emb_0[user_indices]
    pos_emb, pos_emb_0 = items_emb[pos_indices], items_emb_0[pos_indices]
    neg_emb, neg_emb_0 = items_emb[neg_indices], items_emb_0[neg_indices]

    loss = bpr_loss(users_emb, users_emb_0, pos_emb, pos_emb_0,
                    neg_emb, neg_emb_0, lambda_val).item()

    users_emb_w = model.users_emb.weight
    items_emb_w = model.items_emb.weight

    # set interaction matrix between every user and item, mask out existing ones
    interaction_ = torch.matmul(users_emb_w, items_emb_w.T)

    for index in mask_index:
        user_pos_items = get_positive_items(index)
        masked_users = []
        masked_items = []
        for user, items in user_pos_items.items():
            masked_users.extend([user] * len(items))
            masked_items.extend(items)

        interaction_[masked_users, masked_items] = float("-inf")

    _, top_K_items = torch.topk(interaction_, k=k)

    # get all unique users and actual ratings for evaluation
    users = edge_index[0].unique()
    test_user_pos_items = get_positive_items(edge_index)

    actual_r = [test_user_pos_items[user.item()] for user in users]
    pred_r = []

    for user in users:
        items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in items, top_K_items[user]))
        pred_r.append(label)
    
    pred_r = torch.Tensor(np.array(pred_r).astype('float'))
    

    correct_count = torch.sum(pred_r, dim=-1)
    # number of items liked by each user in the test set
    liked_count = torch.Tensor([len(actual_r[i]) for i in range(len(actual_r))])
    
    recall = torch.mean(correct_count / liked_count)
    precision = torch.mean(correct_count) / k


    return loss, recall, precision, top_K_items, users

def recallAtK(actual_r, pred_r, k):
    """
    Return recall at k and precision at k
    """
    assert pred_r is not None,\
        "Wrong Handling on the dataset"
        
    correct_count = torch.sum(pred_r, dim=-1)
    # number of items liked by each user in the test set
    liked_count = torch.Tensor([len(actual_r[i]) for i in range(len(actual_r))])
    
    recall = torch.mean(correct_count / liked_count)
    precision = torch.mean(correct_count) / k
    
    return recall.item(), precision.item()

# model configurations
CFG = {
    'batch_size': 8192,
    'num_epoch': 30,
    'lr': 1e-4,
    'lr_decay': 0.5,
    'topK': 10,
    'lambda': 1e-7,
    # embedding dimension
    'hidden_dim': 128,
    'num_layer': 3,
}

# # setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = LightGCN(preparation.num_users, preparation.num_items, CFG['hidden_dim'], CFG['num_layer'])
model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=CFG['lr_decay'])

edge_index = preparation.edge_index.to(device)
train_edge_index = preparation.train_edge_idx.to(device)
train_sparse_edge_index = preparation.train_sparse_edge_idx.to(device)
print(len(train_edge_index[0]))

val_edge_index = preparation.valid_edge_idx.to(device)
val_sparse_edge_index = preparation.valid_sparse_edge_idx.to(device)

# training loop
train_losses = []

for epoch in tqdm(range(CFG['num_epoch'])):
    for _ in tqdm(range(2000)):
        users_emb, users_emb_0, items_emb, items_emb_0 = \
            model.forward(train_edge_index)

        # mini batching
        user_indices, pos_indices, neg_indices = \
            preparation.batch_sample(CFG['batch_size'], train_edge_index)
        
        user_indices = user_indices.to(device)
        pos_indices = pos_indices.to(device)
        neg_indices = neg_indices.to(device)
        
        users_emb, users_emb_0 = users_emb[user_indices], users_emb_0[user_indices]
        pos_emb, pos_emb_0 = items_emb[pos_indices], items_emb_0[pos_indices]
        neg_emb, neg_emb_0 = items_emb[neg_indices], items_emb_0[neg_indices]
        
        # loss computation
        loss = bpr_loss(users_emb, users_emb_0, 
                        pos_emb, pos_emb_0,
                        neg_emb, neg_emb_0,
                        CFG['lambda'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch {:d}: train_loss: {:.4f}'\
        .format(epoch, loss))
    train_losses.append(loss.item())
    torch.save(model, "model.pt")
    scheduler.step()

# evaluate on test set
with torch.no_grad():
    model = torch.load("model.pt")
    model.eval()
    test_sparse_edge_index = preparation.valid_sparse_edge_idx.to(device)
    test_edge_index = preparation.valid_edge_idx.to(device)
    test_loss, test_recall, test_precision, top_k, users \
        = evaluation(model, 
                    test_edge_index, 
                    test_sparse_edge_index, 
                    [],
                    CFG['topK'],
                    CFG['lambda'])
    

print('Test set: train_loss: {:.4f}, recall: {:.4f}, precision: {:.4f}'\
        .format(test_loss, test_recall, test_precision))