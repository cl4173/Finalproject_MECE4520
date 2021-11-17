# Import libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes.base import ensure_index
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

# Import data & clean
URL=r'C:\Users\liuch\Desktop\Courses\Data_Science_MECE4520\Project\diabetes_012_health_indicators_BRFSS2015.csv'
I_Data=pd.read_csv(URL)
I_Data.fillna(0)
I_Data_S=pd.DataFrame(I_Data[~I_Data["Diabetes_012"].isin([1])],columns=I_Data.columns)# Delete prediabetes(1)
for e in range(len(I_Data_S)):              # One-hot result Y
      if I_Data_S.iloc[e]["Diabetes_012"]!=0:
         I_Data_S.iloc[e]["Diabetes_012"]=1         
Data_1_Y=I_Data_S["Diabetes_012"] # Get Y value
Data_1_X=I_Data_S.drop(["Diabetes_012"],axis=1)# Get X value
S_H=StandardScaler()
SData_1_X=S_H.fit_transform(Data_1_X)# normalize data

#Hyper turning with Assumption 1: delete prediabetes(1)
S_H=StandardScaler()
Data_1=S_H.fit_transform(Data_1_X)# normalize data
H_Parameter=pd.DataFrame(columns=["trial", "C", "train", "validation",'time']) #Define output
H_trails=10 # Hyper turning loop
Test_ratio=0.2 # Define CV test dataset ratio
H_all_models = []
for t in range(H_trails):
    start=time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(SData_1_X,Data_1_Y, test_size=Test_ratio, random_state=(19))
    H_model=sklearn.linear_model.LogisticRegression(penalty="l2",dual=False,max_iter=200,C=1+100*t,solver='sag',n_jobs=-1) # Logistic regression
    H_model.fit(X_train,y_train)# Train model
    Y_predict_T=H_model.predict_proba(X_train) # Predict value of train dataset
    Y_predict_V=H_model.predict_proba(X_validation)# Predict value of validation dataset
    Y_los_T=sklearn.metrics.log_loss(y_train,Y_predict_T)# calculate los for train sets
    Y_los_V=sklearn.metrics.log_loss(y_validation,Y_predict_V)# calculate los for validation sets
    end=time.time()
    Input_row={"trial":t+1, "C":1+100*t, "train":Y_los_T, "validation":Y_los_V,'time':str(end-start)}#Collect input row
    H_Parameter=H_Parameter.append(Input_row,ignore_index=True)#Collect input information into table
    H_all_models.append(H_model)

#draw figure of hyper turning results
print(H_Parameter)
sns.lineplot(x="C", y="train", data=H_Parameter, label="train")
sns.lineplot(x="C", y="validation", data=H_Parameter, label="validation")
plt.gca().set_ylabel("Los & time")
plt.tight_layout()
plt.show()
        