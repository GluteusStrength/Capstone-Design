import pandas as pd
import torch
from preparation import user_map, item_map, get_positive_items, edge_index, patron_dic
from model import LightGCN

model = torch.load("model.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def recommendations(user_id, pos_item):
    swap_user_map = {v: k for k, v in patron_dic.items()}
    swap_item_map = {v: k for k, v in item_map.items()}
    patron_info = pd.concat([pd.read_csv("patron_erica.csv"), pd.read_csv("patron_erica_before.csv")], axis = 0)
    item_info = pd.concat([pd.read_csv("item_erica.csv"), pd.read_csv("item_erica_before.csv")], axis = 0)
    patron_info.drop_duplicates(["ID"], inplace = True, ignore_index = True)
    item_info.drop_duplicates(["TITLE"], inplace = True, ignore_index = True)
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
        rent.append(item_info[item_info["ID"] == swap_item_map[i]].loc[:, "TITLE"].iloc[0])
    rent = list(set(rent))
    for i in rent:
        print(i, end = " | ")
    print()
    print("----------------------------------------------")
    print("Recommendation on user {}".format(user))
    for idx in indices:
        recommend.append(item_info[item_info["ID"] == swap_item_map[idx]].loc[:, "TITLE"].iloc[0])
    recommend = list(set(recommend))
    for i in recommend:
        print(i, end = " | ")
    print("\n")

# pos_items.keys()[123]
pos_items = get_positive_items(edge_index)
recommendations(user_id = list(pos_items.keys())[331], pos_item = pos_items)