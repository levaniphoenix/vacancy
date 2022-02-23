import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

df= pd.read_csv("data set(corelation filtered).csv",na_values="NA")
y=df["SalePrice"]
y=(y >= y.mean()).astype(int)
df.drop(["SalePrice"],axis=1,inplace=True)
x= df.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()

X_train_scaled = preprocessing.StandardScaler().fit(X_train).transform(X_train)
X_test_scaled = preprocessing.StandardScaler().fit(X_test).transform(X_test)

model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)

print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))