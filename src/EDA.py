import pandas as pd
import numpy as np


# adsData = pd.read_csv("../data/t_ads.csv")
# commentData = pd.read_csv("../data/t_comment.csv")
orderData = pd.read_csv("../data/t_order.csv")
#productData = pd.read_csv("../data/t_product.csv")
saleData = pd.read_csv("../data/t_sales_sum.csv")
saleData["date"] = pd.to_datetime(saleData["dt"])
orderData["date"] = pd.to_datetime(orderData['ord_dt']) # 解析日期
del orderData['ord_dt']
orderGroup = orderData.groupby('shop_id', as_index=True).resample('1D', on='date') #按天聚合

# 日加和
orderDataSum = orderGroup.sum()
del orderDataSum["shop_id"]
orderDataSum.reset_index(level=['shop_id', 'date'], inplace=True)
orderDataSum["month"] = orderDataSum["date"].apply(lambda x:x.month)

shopData = orderDataSum[orderDataSum["shop_id"] == 1]
del shopData["shop_id"]
del shopData["pid"]
augstLabel = shopData[(shopData["month"] >= 9) & (shopData["month"] <= 11)].sum()
print(augstLabel)

print(saleData[(saleData["date"] == "2016-08-31")&(saleData["shop_id"] == 1)]["sale_amt_3m"])



# cateNum = productData.groupby(["shop_id"])["cate"].value_counts().to_frame('cate_num')
#
# cateNum.reset_index(level=['shop_id', 'cate'], inplace=True)
# cateNum = cateNum.groupby("shop_id").max()
# cateNum.reset_index(inplace=True)
# cateNum.rename(columns={"cate":"max_cate"}, inplace=True)
# print(cateNum)





# print(adsData.head(5))
# print(commentData.head(5))
# print(orderData.head(5))
# print(productData.head(5))
# print(saleData.head(5))
#
# print(len(adsData))
# print(len(commentData))
# print(len(orderData))
# print(len(productData))
# print(len(saleData))

# totaoID = pd.concat([adsData["shop_id"], commentData["shop_id"], orderData["shop_id"]
#                     ,productData["shop_id"], saleData["shop_id"]])
#
# print(len(totaoID.drop_duplicates()))
# print(len(adsData["shop_id"].drop_duplicates()))
# print(len(commentData))
# print(len(orderData))
# print(len(productData))
#print(len(saleData)) # 24030