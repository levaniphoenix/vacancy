import pandas as pd
import numpy as np


def preprocessData():
    df= pd.read_csv("data set.csv",na_values="NA")
    df=df.drop(['Id'],axis=1)

    types=df.dtypes
    colnames=[]
    for i in range(0,len(types)):
        if types[i] == "object":
            colnames.append( df.columns[i])
        elif types[i] == "float64" or types[i] == "int64":
            colname=df.columns[i]
            df[colname]=df[colname].replace(np.nan,df[colname].mean())

    df=df.replace(np.nan, "U")
    df=pd.get_dummies(df,columns=["MSSubClass"])
    df=pd.get_dummies(df,columns=colnames)
    df.to_csv("data set(encoded).csv",sep=",",index=False)

    corelation=df.corr()
    colnames=[]
    for i in range(0,df.shape[1]):
        if abs(corelation['SalePrice'][i]) >= 0.5:
            colnames.append(df.columns[i])
    df[colnames].to_csv("data set(corelation filtered).csv",sep=",",index=False)

preprocessData()
