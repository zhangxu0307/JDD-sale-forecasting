import pandas as pd

saleData = pd.read_csv("../data/t_sales_sum.csv")
saleData["date"] = pd.to_datetime(saleData['dt'])
meanData = saleData.groupby("shop_id").mean().reset_index()
print(meanData)
meanData.to_csv("../result_mean.csv", index=False)
