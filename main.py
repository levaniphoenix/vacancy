import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df= pd.read_csv("data set(corelation filtered).csv",na_values="NA")
y=df["SalePrice"].to_numpy()
df.drop(["SalePrice"],axis=1,inplace=True)
x= df.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)

model = linear_model.LinearRegression()
reg = linear_model.Lasso(alpha=0.1)

model.fit(X_train, Y_train)
reg.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
Y_pred_lasso=reg.predict(X_test)

print('Mean squared error (MSE) of Ordinary Least Squares: %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2) of Ordinary Least Squares: %.2f'
      % r2_score(Y_test, Y_pred))

print('Mean squared error (MSE) of Lasso: %.2f'
      % mean_squared_error(Y_test, Y_pred_lasso))
print('Coefficient of determination (R^2) of Lasso: %.2f'
      % r2_score(Y_test, Y_pred_lasso))