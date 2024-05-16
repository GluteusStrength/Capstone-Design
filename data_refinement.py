# 데이터셋 정제 작업
# 이용자 정보와 도서 정보에 대한 적절한 Representation을 들어가기 전의 작업
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

data = pd.read_excel("library.xlsx", sheet_name = None)

"""
charge -> 대출 ID, 사용자 ID, 도서 ID 정보 이용
patron -> 사용자 ID, 이용자 정보 ID 값, 소속명, 성별, 생년월일
item -> 도서 ID, 책 제목, 저자, Class 정보
hold -> 예약 정보
"""
charge = data["charge(대출)"]
patron = data["patron(이용자)"]
item = data["item(도서)"]
hold = data["hold(예약)"]

patron["BIRTHDAY"] //= 10000
# 학교 단과대학 기준으로 우선 데이터 분리
patron_erica = patron[(patron["상위소속명"] == "예체능대학") | (patron["상위소속명"] == "공학대학") 
       | (patron["상위소속명"] == "소프트웨어융합대학") | (patron["상위소속명"] == "과학기술융합대학")
       | (patron["상위소속명"] == "국제문화대학") | (patron["상위소속명"] == "디자인대학")
       | (patron["상위소속명"] == "약학대학") | (patron["상위소속명"] == "언론정보대학") 
       | (patron["상위소속명"] == "경상대학")]

patron_erica.reset_index(inplace=True)
patron_erica.drop(['index'], axis = 1, inplace = True)

# PATRON_ID: 이용자 ID
rent = np.array(charge["PATRON_ID"])
# ERICA 이용자 ID
id_ = np.array(patron_erica["ID"])
index_ = []
users = []
for idx in range(len(rent)):
    if rent[idx] in id_:
        index_.append(idx)
        users.append(rent[idx])
charge_erica = charge.loc[index_, :]
# 실제로 대출을 1회라도 한 ID 제출.
print("{}명의 사용자가 대출".format(len(np.unique(users))))

charge_item = np.unique(charge_erica["ITEM_ID"])
item_idx = [idx for idx in range(len(item)) if item.loc[idx, "ID"] in charge_item]
item_erica = item.loc[item_idx, :]
# 에리카 학생들이 대여한 책들
item_erica.reset_index(inplace = True)
item_erica.drop(["index"], axis = 1, inplace = True)

class_no = item_erica["CLASS_NO"]
classes = []
for i in class_no:
    if i[0] == " ":
        classes.append(i[1])
    else:
        classes.append(i[0])
item_erica["target"] = classes

# 우선 예약 횟수에 대한 정보를 hold라는 새로운 column으로 추가한다.
hold_dict = {}
hold_items = hold["ITEM_ID"].value_counts()
indexes = hold_items.index
for ids in range(len(hold_items)):
    hold_dict[indexes[ids]] = hold_items[indexes[ids]]

action = []
for i in range(len(item_erica)):
    ID = item["ID"][i]
    if ID not in list(hold_dict.keys()):
        action.append(0)
    else:
        action.append(hold_dict[ID])
item_erica["hold"] = action

item_erica["charge"] = [0]*len(item_erica)
from tqdm import tqdm
id_list = list(charge["ITEM_ID"])
ids = dict()
for i in tqdm(id_list):
    if i not in list(ids.keys()):
        ids[i] = 1
    else:
        ids[i] += 1

for i in tqdm(range(len(item_erica))):
    item_erica.loc[i, "charge"] = ids[item_erica.loc[i, "ID"]]

charge.to_csv("charge.csv", index = False)
item_erica.to_csv("item_erica.csv", index = False)
patron.to_csv("patron_erica.csv", index = False)