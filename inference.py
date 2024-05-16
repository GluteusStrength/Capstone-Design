import pandas as pd
import torch
from preparation import user_map, item_map, get_positive_items, edge_index
from model import LightGCN

model = torch.load("model.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def recommendations(user_id, pos_item):
    swap_user_map = {v: k for k, v in user_map.items()}
    swap_item_map = {v: k for k, v in item_map.items()}
    patron_info = pd.read_csv("patron_erica.csv")
    item_info = pd.read_csv("item_erica.csv")
    user = swap_user_map[user_id]
    user_emb = model.users_emb.weight[user_id]
    scores = model.items_emb.weight @ user_emb
    _, indices = torch.topk(scores, k = 10)
    indices = indices.detach().cpu().numpy()
    print(patron_info[patron_info["ID"] == user])
    print("----------------------------------------------")
    print("{} liked these books".format(user))
    rent = []
    recommend = []
    for i in pos_item[user_id]:
        rent.append(item_info[item_info["ID"] == swap_item_map[i]].loc[:, "TITLE"])
    print(rent)
    print("----------------------------------------------")
    print("Recommendation on user {}".format(user))
    for idx in indices:
        recommend.append(item_info[item_info["ID"] == swap_item_map[idx]].loc[:, "TITLE"])
    print(recommend)

pos_items = get_positive_items(edge_index)
recommendations(user_id = list(pos_items.keys())[231], pos_item = pos_items)