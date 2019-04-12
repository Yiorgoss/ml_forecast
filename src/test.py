import pandas as pd 
import numpy as np

print("STRT")
data = pd.read_csv("/dltraining/datasets/single_file.csv", infer_datetime_format=True, index_col=[0])
print(data['aapl'].isna().sum())
print(type(data))

print(data['aapl'])
data.dropna(axis=1, inplace=True)
data.to_csv("/dltraining/datasets/single_file.csv")
print("DNE")
