# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # <b> <font color=orange> Hồi quy tuyến tính (Linear Regression)
# ---
# </font>

# %%
import pandas as pd

from model.LinearRegression import *
from utils.Functions import *


# %%
df_luong = pd.read_csv('dataset/luong.csv')
df_luong


# %%
# Tự xây dựng mô hình
model = LinearRegression()
X, y, headers = input_label_split(df=df_luong)
# model.fit_grad(X, y, numOfIteration=None, learning_rate=1e-5, threshhold=1e-3, show_loss=True)
model.fit(X, y)


# %%
# Sử dụng mô hình có sẵn trong thư viện scikit-learn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y);


# %%
model.w


# %%
lr.intercept_, lr.coef_


# %%
model.score(X, y)


# %%
lr.score(X, y)


# %%
model.show(X, y, labels=headers)


# %%
df_luong2 = pd.read_csv('dataset/luong2.csv')
df_luong2['Bằng cấp chuyên môn'].replace({'No': 0, 'Yes': 1}, inplace = True)
df_luong2


# %%
X, y, _ = input_label_split(df=df_luong2)
# model.fit_grad(X, y, numOfIteration=None, learning_rate=1e-5, threshhold=1e-3, show_loss=True)
model.fit(X, y)
lr.fit(X, y);


# %%
model.w


# %%
lr.intercept_, lr.coef_


# %%
model.score(X, y)


# %%
lr.fit(X, y)
lr.score(X, y)


