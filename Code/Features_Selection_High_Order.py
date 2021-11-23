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
URL=r'C:\Users\liuch\Desktop\Courses\Data_Science_MECE4520\Project\diabetes_binary_health_indicators_BRFSS2015.csv'
I_Data=pd.read_csv(URL)
I_Data.fillna(0)
I_Data_S=pd.DataFrame(I_Data,columns=I_Data.columns)
print(I_Data_S.shape)
for e in range(len(I_Data_S)):              # standlize result Y
      if I_Data_S.iloc[e]["Diabetes_binary"]!=0:
         I_Data_S.iloc[e]["Diabetes_binary"]=1         
Data_1_Y=I_Data_S["Diabetes_binary"] # Get Y value
Data_1_X=I_Data_S.drop(["Diabetes_binary"],axis=1)# Get X value
# Add additional features
Order=8 # Define maximum
H_Parameter=pd.DataFrame(columns=["trial", "C", "train", "validation",'time']) #Define output
for o in range(2,Order+2):
    Data_1_X[f"BMI{o}"] =Data_1_X["BMI"].apply(lambda x: x**o)
    Data_1_X[f"GenHlth{o}"] =Data_1_X["GenHlth"].apply(lambda x: x**o)
    Data_1_X[f"MentHlth{o}"] = Data_1_X["MentHlth"].apply(lambda x: x**o)
    Data_1_X[f"PhysHlth{o}"] = Data_1_X["PhysHlth"].apply(lambda x: x**o)
    Data_1_X[f"Age{o}"] = Data_1_X["Age"].apply(lambda x: x**o)
    Data_1_X[f"Education{o}"] = Data_1_X["Education"].apply(lambda x: x**o)
    Data_1_X[f"Income{o}"] = Data_1_X["Income"].apply(lambda x: x**o)
    S_H=StandardScaler()
    SData_1_X=S_H.fit_transform(Data_1_X)# normalize data
    Test_ratio=0.2 # Define CV test dataset ratio
    H_all_models = []
    start=time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(SData_1_X, Data_1_Y, test_size=Test_ratio, random_state=(19))
    H_model=sklearn.linear_model.LogisticRegression(penalty='elasticnet',dual=False,solver='saga',C=0.05,tol=1e-5 ,l1_ratio=0.5,max_iter=22000,n_jobs=-1) # Logistic regression
    H_model.fit(X_train,y_train)# Train model
    Y_predict_T=H_model.predict_proba(X_train) # Predict value of train dataset
    Y_predict_V=H_model.predict_proba(X_validation)# Predict value of validation dataset
    Y_los_T=sklearn.metrics.log_loss(y_train,Y_predict_T)# calculate los for train sets
    Y_los_V=sklearn.metrics.log_loss(y_validation,Y_predict_V)# calculate los for validation sets
    end=time.time()
    Input_row={"trail": o,"order":o, "C":"None", "train":Y_los_T, "validation":Y_los_V,'time':str(end-start)}#Collect input row
    H_Parameter=H_Parameter.append(Input_row,ignore_index=True)#Collect input information into table
    print(o)
    H_all_models.append(H_model)

#draw figure of hyper turning results
print(H_Parameter)
AA=plt.figure()
PIC1=AA.add_subplot(1,2,1)
sns.lineplot(x="order", y="train", data=H_Parameter, label="train")
sns.lineplot(x="order", y="validation", data=H_Parameter, label="validation")
plt.gca().set_ylabel("Los")
plt.title(f'Order & Los')
PIC2=AA.add_subplot(1,2,2)
sns.lineplot(x="order", y="time", data=H_Parameter, label="train")
sns.lineplot(x="order", y="time", data=H_Parameter, label="validation")
plt.gca().set_ylabel("Order & Time")
plt.tight_layout()
plt.show()
