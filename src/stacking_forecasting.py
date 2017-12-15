import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import lightgbm as lgb
import matplotlib as mpl
mpl.use('Agg')
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn import model_selection


def WMAE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred) )/ np.sum(y_true)

# 处理order表
orderData = pd.read_csv("../data/t_order.csv")
orderData["date"] = pd.to_datetime(orderData['ord_dt']) # 解析日期
del orderData['ord_dt']

# pid数目统计
#pnum = orderData[["shop_id", "pid", "date"]].groupby(["shop_id", "date"], as_index=False).count().rename(columns={"pid":"pidnum"})
#pUniqueNum = orderData[["shop_id", "pid", "date"]].groupby(["shop_id", "date"], as_index=False).nunique().rename(columns={"pid":"pid_unique_num"})
del orderData["pid"]

orderGroup = orderData.groupby('shop_id', as_index=True).resample('1M', on='date') #按月聚合

# 月加和
orderDataSum = orderGroup.sum()
del orderDataSum["shop_id"]
orderDataSum.reset_index(level=['shop_id', 'date'], inplace=True)

order = orderDataSum
#order = pd.merge(order, pnum, how="left", on=["shop_id", "date"])
#order = pd.merge(order, pUniqueNum, how="left", on=["shop_id", "date"])


order = order.fillna(order.mean()) # 填补空值
print("order data features:", order.columns)
print("order data num:", len(order))
print("order table finished!")

# 处理评论表
commentData = pd.read_csv("../data/t_comment.csv")
commentData["date"] = pd.to_datetime(commentData['create_dt']) # 解析日期
del commentData['create_dt']

commentDataGroup = commentData.groupby('shop_id', as_index=True).resample('1M', on='date')  #按月聚合

# 月加和
commentDataSum = commentDataGroup.sum()
del commentDataSum["shop_id"]
commentDataSum.reset_index(level=['shop_id', 'date'], inplace=True)

comment = commentDataSum

comment = comment.fillna(comment.mean()) # 填补空值
print("comment data features:", comment.columns)
print("comment data num:", len(comment))
print("comment tabel finished!")

# 处理广告表
adsData = pd.read_csv("../data/t_ads.csv")
adsData["date"] = pd.to_datetime(adsData['create_dt']) # 解析日期
del adsData['create_dt']

ads = adsData.groupby('shop_id', as_index=True).resample('1M', on='date').sum() #按月聚合
del ads["shop_id"]
ads.reset_index(level=['shop_id', 'date'], inplace=True)
ads = ads.fillna(ads.mean())
print("ads data features:", ads.columns)
print("ads data num:", len(ads))
print("ads table finished!")

# 处理商品表
productData = pd.read_csv("../data/t_product.csv")
distinctNum = productData.groupby(by=["shop_id"])["brand", "cate"].nunique() # 统计品牌数目和种类数目
distinctNum.reset_index(inplace=True)

# 统计每个shop每月的上货数目
productData["date"] = pd.to_datetime(productData['on_dt']) # 解析日期
productMonth_on = productData.groupby('shop_id', as_index=True)["date", "pid"].resample('1M', on='date').count()
del productMonth_on["date"]
productMonth_on = productMonth_on.reset_index(level=["shop_id", "date"]).rename(columns={"pid":"on_num"})

# 统计下货的数目，商品下货日期均为5月1日，因此这是个shop的静态特征
productData = productData.dropna()
productData["date"] = pd.to_datetime(productData['off_dt']) # 解析日期
productMonth_off = pd.DataFrame(np.arange(1,3001), index=np.arange(1,3001), columns=["off_num"])
productMonth_off.index.name="shop_id"
productMonth_off["off_num"] = 0
productMonth_off_temp = productData.groupby('shop_id', as_index=True)["date", "pid"].resample('1M', on='date').count()
del productMonth_off_temp["date"]
productMonth_off_temp = productMonth_off_temp.reset_index(level=["date"]).rename(columns={"pid" : "off_num"})
del productMonth_off_temp["date"]
productMonth_off += productMonth_off_temp
productMonth_off.fillna(0, inplace=True)
productMonth_off.reset_index(inplace=True)

# 统计shop中最多的brand的编码和数量
brandNum = productData.groupby(["shop_id"])["brand"].value_counts().to_frame('brand_num')
brandNum.reset_index(level=['shop_id', 'brand'], inplace=True)
brandNum = brandNum.groupby("shop_id").max()
brandNum.reset_index(inplace=True)
brandNum.rename(columns={"brand":"max_brand"}, inplace=True)


# 统计shop中最多的cate的编码和数量
cateNum = productData.groupby(["shop_id"])["cate"].value_counts().to_frame('cate_num')
cateNum.reset_index(level=['shop_id', 'cate'], inplace=True)
cateNum = cateNum.groupby("shop_id").max()
cateNum.reset_index(inplace=True)
cateNum.rename(columns={"cate":"max_cate"}, inplace=True)

print("product data finished!")

# 链接所有表
totalData = pd.merge(order, comment, on=["shop_id", "date"], how="outer")
totalData = pd.merge(totalData, adsData, on=["shop_id", "date"], how="outer")
totalData = pd.merge(totalData, distinctNum, on="shop_id", how="left")
totalData = pd.merge(totalData, productMonth_on, on=["shop_id", "date"], how="left")
totalData = pd.merge(totalData, productMonth_off, on="shop_id", how="left")
totalData = pd.merge(totalData, brandNum, on="shop_id", how="left")
totalData = pd.merge(totalData, cateNum, on="shop_id", how="left")

# 提取月份特征
totalData["month"] = totalData["date"].apply(lambda x: x.month)
print("total features:", totalData.columns)
print("total data num:", len(totalData))

# 链接销量表作为label
saleData = pd.read_csv("../data/t_sales_sum.csv")
saleData["date"] = pd.to_datetime(saleData['dt'])
del saleData["dt"]
train = pd.merge(saleData, totalData, on=["shop_id", "date"], how="left") # 此处左连接，表示只有saledata中八个月的数据作为训练
print("total data shop id num:", len(totalData["shop_id"].drop_duplicates()))
train = train.fillna(-1)

# 整理train和test以及submit
feature_name = [ 'shop_id',  'sale_amt', 'offer_amt', 'offer_cnt',
       'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt',
        #'pidnum',
        # 'pid_unique_num',
       'bad_num',
       'cmmt_num', 'dis_num', 'good_num', 'mid_num',
        'charge', 'consume',
        'brand', 'cate',
       'on_num', 'off_num', "brand_num", "cate_num", "max_cate", "max_brand"
        #'month'
                 ]
# feature_name = ['shop_id', 'sale_amt', 'offer_amt', 'offer_cnt',
#        'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt',
#        'sale_amt_mean', 'offer_amt_mean', 'offer_cnt_mean', 'rtn_cnt_mean',
#        'rtn_amt_mean', 'ord_cnt_mean', 'user_cnt_mean',
#        'sale_amt_min', 'offer_amt_min', 'offer_cnt_min', 'rtn_cnt_min',
#        'rtn_amt_min', 'ord_cnt_min', 'user_cnt_min', 'sale_amt_std',
#        'offer_amt_std', 'offer_cnt_std', 'rtn_cnt_std', 'rtn_amt_std',
#        'ord_cnt_std', 'user_cnt_std', 'sale_amt_median', 'offer_amt_median',
#        'offer_cnt_median', 'rtn_cnt_median', 'rtn_amt_median',
#        'ord_cnt_median', 'user_cnt_median', 'bad_num', 'cmmt_num',
#        'dis_num', 'good_num', 'mid_num', 'bad_num_mean',
#        'cmmt_num_mean', 'dis_num_mean', 'good_num_mean', 'mid_num_mean', 'bad_num_min',
#        'cmmt_num_min', 'dis_num_min', 'good_num_min', 'mid_num_min',
#        'bad_num_median', 'cmmt_num_median', 'dis_num_median',
#        'good_num_median', 'mid_num_median', 'charge', 'consume', 'brand',
#        'cate', 'on_num', 'month']

trainx = train[feature_name]
trainy = train["sale_amt_3m"]

testx = totalData[totalData["date"] == "2017-04-30"] # 4月份的数据作为测试集
testx = testx.fillna(-1)
testx = testx[feature_name]

#trainy = np.log1p(trainy)


# 网格搜索

print("网格搜索开始.....")
modelNum = 4

lossFunc = make_scorer(WMAE, greater_is_better=False)
model1 = xgb.XGBRegressor(silent=True)
model2 = lgb.LGBMRegressor(silent=True)
model3 = RandomForestRegressor()
model4 = GradientBoostingRegressor()
modelList = [model1, model2, model3, model4]

params1 = {"learning_rate": [0.5, 0.1, 0.01], "max_depth": [10, 15, 20, 25], "n_estimators":[100, 150, 200, 250]}
params2 = {"learning_rate": [0.5, 0.1, 0.01], "max_depth": [10, 15, 20, 25], "n_estimators":[100, 150, 200, 250]}
params3 = {"max_depth": [10, 15, 20, 25, 30], "n_estimators":[50, 100, 150, 200, 250]}
params4 = {"learning_rate": [0.5, 0.1, 0.01], "max_depth": [10, 15, 20, 25], "n_estimators":[100, 150, 200, 250]}
paramList = [params1, params2, params3, params4]

for i in range(modelNum):

    model = modelList[i]
    params = paramList[i]
    gsearch = grid_search.GridSearchCV(estimator=model, param_grid=params, scoring=lossFunc, cv=5)
    gsearch.fit(trainx, trainy)
    print("best param:", gsearch.best_params_)
    print("best score:", gsearch.best_score_)








