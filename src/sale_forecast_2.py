import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import xgboost as xgb
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

def WMAE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred) )/ np.sum(y_true)

def slidWindowTrainData(trainFlag):

    # 处理order表
    orderData = pd.read_csv("../data/t_order.csv")
    orderData["date"] = pd.to_datetime(orderData['ord_dt']) # 解析日期
    del orderData['ord_dt']
    del orderData["pid"]

    orderGroup = orderData.groupby('shop_id', as_index=True).resample('1D', on='date') #按天聚合

    # 日加和
    orderDataSum = orderGroup.sum()
    del orderDataSum["shop_id"]
    orderDataSum.reset_index(level=['shop_id', 'date'], inplace=True)

    order = orderDataSum

    order = order.fillna(value=0) # 填补空值
    print("order data features:", order.columns)
    print("order data num:", len(order))
    print("order table finished!")


    # 处理评论表
    commentData = pd.read_csv("../data/t_comment.csv")
    commentData["date"] = pd.to_datetime(commentData['create_dt']) # 解析日期
    del commentData['create_dt']

    commentDataGroup = commentData.groupby('shop_id', as_index=True).resample('1D', on='date')  #按天聚合

    # 日加和
    commentDataSum = commentDataGroup.sum()
    del commentDataSum["shop_id"]
    commentDataSum.reset_index(level=['shop_id', 'date'], inplace=True)

    #commentDataNorm = commentDataSum/orderData["ord_cnt"]

    comment = commentDataSum
    comment = comment.fillna(value=0) # 填补空值

    print("comment data features:", comment.columns)
    print("comment data num:", len(comment))
    print("comment tabel finished!")


    # 处理广告表
    adsData = pd.read_csv("../data/t_ads.csv")
    adsData["date"] = pd.to_datetime(adsData['create_dt']) # 解析日期
    del adsData['create_dt']

    ads = adsData.groupby('shop_id', as_index=True).resample('1D', on='date').sum() #按天聚合
    del ads["shop_id"]
    ads.reset_index(level=['shop_id', 'date'], inplace=True)
    ads = ads.fillna(value=0)

    print("ads data features:", ads.columns)
    print("ads data num:", len(ads))
    print("ads table finished!")

    #处理商品表
    productData = pd.read_csv("../data/t_product.csv")
    distinctNum = productData.groupby(by=["shop_id"])["brand", "cate"].nunique() # 统计品牌数目和种类数目
    distinctNum.reset_index(inplace=True)

    # 处理商品表静态特征，与时间无关
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


    # 处理商品表动态特征，与时间有关

    # 统计每个shop每天的上货数目
    productData["date"] = pd.to_datetime(productData['on_dt']) # 解析日期
    productMonth_on = productData.groupby('shop_id', as_index=True)["date", "pid"].resample('1D', on='date').count()
    del productMonth_on["date"]
    productMonth_on = productMonth_on.reset_index(level=["shop_id", "date"]).rename(columns={"pid":"on_num"})

    # 统计下货的数目，商品下货日期均为5月1日，因此这是个shop的静态特征
    productData = productData.dropna()
    productData["date"] = pd.to_datetime(productData['off_dt']) # 解析日期
    productMonth_off = pd.DataFrame(np.arange(1,3001), index=np.arange(1, 3001), columns=["off_num"])
    productMonth_off.index.name="shop_id"
    productMonth_off["off_num"] = 0
    productMonth_off_temp = productData.groupby('shop_id', as_index=True)["date", "pid"].resample('1M', on='date').count()
    del productMonth_off_temp["date"]
    productMonth_off_temp = productMonth_off_temp.reset_index(level=["date"]).rename(columns={"pid" : "off_num"})
    del productMonth_off_temp["date"]
    productMonth_off += productMonth_off_temp
    productMonth_off.fillna(0, inplace=True)
    productMonth_off.reset_index(inplace=True)

    print("product data finished!")


    # 链接各表
    totalData = pd.merge(order, comment, on=["shop_id", "date"], how="left")
    totalData = pd.merge(totalData, ads, on=["shop_id", "date"], how="left")
    totalData = pd.merge(totalData, productMonth_on, on=["shop_id", "date"], how="left")
    totalData.fillna(value=0, inplace=True)
    print("total data num:", len(totalData))
    #print(totalData)
    testData = totalData[totalData["date"] >= "2017-04-01"] # 4月份的数据作为测试集

    if trainFlag:
        # 滑动窗口采集样本
        trainData = pd.DataFrame()
        trainy = []

        trainSpan = 30
        slidWindowSize = 90
        totalDays = 270
        sampleStep = 1
        trainGroup = totalData.groupby(by="shop_id")
        #count = 0
        for group in trainGroup:

            #count += 1
            start = 0
            subTrainData = group[1]
            shopid = group[0]
            #subTrainData = subTrainData.reset_index()
            print("shop id:", shopid)
            print("single shop data num:", len(subTrainData))

            while start+trainSpan+slidWindowSize < len(subTrainData):

                end = start + trainSpan
                end2 = end + slidWindowSize

                x = subTrainData[start:end].sum()
                x["shop_id"] = shopid
                #print(subTrainData[start:end].iloc[-1]["date"])
                x["date"] = subTrainData[start:end].iloc[-1]["date"]
                #print(x)
                trainData = trainData.append(x, ignore_index=True)

                y = subTrainData[end:end2]["sale_amt"].sum()
                trainy.append(y)
                start += sampleStep

        # 加label
        trainData["sale_amt_3m"] = trainy

        # 链接商品静态特征
        trainData = pd.merge(trainData, distinctNum, on="shop_id", how="left")
        trainData = pd.merge(trainData, brandNum, on="shop_id", how="left")
        trainData = pd.merge(trainData, cateNum, on="shop_id", how="left")
        trainData = pd.merge(trainData, productMonth_off, on="shop_id", how="left")
        trainData.to_csv("../data/slide_window_samples2.csv", index=False)
        #print(trainData)
        print("trainx features:", trainData.columns)
    else:
        testData = testData.groupby('shop_id', as_index=True).resample('1M', on='date').sum()  # 按月聚合
        del testData["shop_id"]

        testData.reset_index(level=['shop_id', 'date'], inplace=True)
        testData = pd.merge(testData, distinctNum, on="shop_id", how="left")
        testData = pd.merge(testData, brandNum, on="shop_id", how="left")
        testData = pd.merge(testData, cateNum, on="shop_id", how="left")
        testData = pd.merge(testData, productMonth_off, on="shop_id", how="left")
        print(testData.columns)
        testData.to_csv("../data/test.csv", index=False)
        #print(testData)
        print("testx features:", testData.columns)



if __name__ == '__main__':

    slidWindowTrainData(trainFlag=True)

    slidWindowTrainData(trainFlag=False)

    # feature_name = [
    #                 'shop_id', 'sale_amt', 'offer_amt', 'offer_cnt',
    #                 'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt',
    #                 # 'pidnum',
    #                 # 'pid_unique_num',
    #                 'bad_num',
    #                 'cmmt_num', 'dis_num', 'good_num', 'mid_num',
    #                 'charge', 'consume',
    #                 'brand', 'cate',
    #                 #'on_num',
    #                 #'off_num',
    #                 "brand_num", "cate_num", "max_cate", "max_brand",
    #                 # 'month'
    #                 #'date'
    #                 ]
    # data = pd.read_csv("../data/slide_window_samples2.csv")
    # # print(data["date"])
    # # data["date"] = data["date"].apply(lambda x:x[:10])
    # # print(data["date"])
    # # data["date"] = pd.to_datetime(data["date"][:10])
    # # data.reset_index("date", inplace=True)
    # # print(data.index)
    #
    # # 最终训练全集数据
    # datax = data[feature_name]
    # datay = data["sale_amt_3m"]
    #
    # # 测试数据
    # testx = pd.read_csv("../data/test.csv") # 4月份的数据作为测试集
    # testx = testx[feature_name]
    #
    # print("data load finished!")
    #
    # # 提交文件数据
    # submit = pd.DataFrame()
    # submit["shop_id"] = testx["shop_id"]
    #
    # # print("网格搜索")
    # # lossFunc = make_scorer(WMAE, greater_is_better=False)
    # # model = xgb.XGBRegressor(objective="reg:linear")
    # # params = {"learning_rate": [0.5, 0.1, 0.01], "max_depth": [10, 20, 30], "n_estimators":[100,  200, 300]}
    # # gsearch = GridSearchCV(estimator=model, param_grid=params, scoring=lossFunc, cv=3)
    #
    # # gsearch.fit(datax, datay)
    # # print("best param:", gsearch.best_params_)
    # # print("best score:", gsearch.best_score_)
    #
    # # 交叉验证, 按照时间先后顺序划分验证集
    # print("验证")
    # train = data[data["date"] < "2016-12-31"]
    # val = data[data["date"] >= "2016-12-31"]
    # print(train["date"])
    # print("train data num:", len(train))
    # print("val data num:", len(val))
    #
    # trainx = train[feature_name]
    # trainy = train["sale_amt_3m"]
    # valx = val[feature_name]
    # valy = val["sale_amt_3m"]
    # val_sale = val['sale_amt']
    #
    # k = 1
    # scores = []
    # for i in range(k):
    #     model = xgb.XGBRegressor(objective="reg:linear",
    #                              learning_rate= 0.01, #gsearch.best_params_['learning_rate'],
    #                              max_depth= 4,# gsearch.best_params_['max_depth'],
    #                              n_estimators=150 ,#gsearch.best_params_['n_estimators'],
    #                              silent=False,
    #                              colsample_bytree=0.9,
    #                              )
    #     model.fit(trainx, trainy)
    #     pre = model.predict(valx)
    #     score = WMAE(valy, pre)
    #     scores.append(score)
    # print("valid scores:", scores)
    # print("mean score:", np.mean(scores))
    #
    # # 测试
    # print("训练预测")
    #
    # model = xgb.XGBRegressor(objective="reg:linear",
    #                          learning_rate= 0.01, #gsearch.best_params_['learning_rate'],
    #                          max_depth= 5,# gsearch.best_params_['max_depth'],
    #                          n_estimators=200 ,#gsearch.best_params_['n_estimators'],
    #                          silent=True,
    #                          colsample_bytree=0.9,
    #                          )
    # model.fit(trainx, trainy)
    # result = model.predict(testx)
    # submit['prediction'] = result
    # submit.to_csv("../xgb_result_sw.csv", index=False)
    # print("predicting finished!")






