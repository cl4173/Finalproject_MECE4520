# Import libs
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
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

# Import data_A & clean
URL=r'C:\Users\liuch\Desktop\Courses\Data_Science_MECE4520\Project\diabetes_012_health_indicators_BRFSS2015.csv'
I_Data=pd.read_csv(URL)
I_Data.fillna(0)
# Import data_B & clean
URL=r'C:\Users\liuch\Desktop\Courses\Data_Science_MECE4520\Project\diabetes_binary_health_indicators_BRFSS2015.csv'
I_Data_B=pd.read_csv(URL)
I_Data_B=pd.DataFrame(I_Data_B,columns=I_Data_B.columns)
I_Data_B.fillna(0)

# Setup dataset_A
I_Data_A=pd.DataFrame(I_Data[~I_Data["Diabetes_012"].isin([1])],columns=I_Data.columns)# Delete prediabetes(1)
for e in range(len(I_Data_A)):              # standlize result Y
      if I_Data_A.iloc[e]["Diabetes_012"]!=0:
         I_Data_A.iloc[e]["Diabetes_012"]=1     

#Split Data set to training and validations
N_splits=10   #Define K fold 
N_trails=1    #Define loops(random seed)
a=0           #Define F loop number for dataset a
b=0           #Define F loop number for dataset b
H_Parameter_A=pd.DataFrame(columns=["K","train", "validation",'time']) #Define output of A
H_Parameter_B=pd.DataFrame(columns=["K","train", "validation",'time']) #Define output of B
for i in range(N_trails):

    #Train and validate using Dataset A
    KF1=KFold(n_splits=N_splits,shuffle=True,random_state=19+i)   
    for train_index_A, test_index_A in KF1.split(I_Data_A):
        #Normalize data
        a=a+1
        print(f"Dataset_A Loop:{a}")
        y_train_A=I_Data_A.iloc[train_index_A]["Diabetes_012"] # Preserve 0/1
        y_test_A=I_Data_A.iloc[test_index_A]["Diabetes_012"]   # Preserve 0/1
        St_train_1A=StandardScaler()
        St_test_1A=StandardScaler()
        train_1A=pd.DataFrame(St_train_1A.fit_transform(I_Data_A.iloc[train_index_A]),columns=I_Data_A.columns)# Get train dataset
        test_1A=pd.DataFrame(St_test_1A.fit_transform(I_Data_A.iloc[test_index_A]),columns=I_Data_A.columns)   # Get test dataset
        X_tr_A=train_1A.drop(["Diabetes_012"],axis=1) # Get X train
        X_te_A=test_1A.drop(["Diabetes_012"],axis=1)   # Get X test     
        # Fit model A
        H_all_models_A = []
        #Start fit process
        start_A=time.time()
        H_model_A=sklearn.linear_model.LogisticRegression(penalty='none',dual=False,solver='sag',max_iter=8000) # Logistic regression
        H_model_A.fit(X_tr_A,y_train_A)# Train model
        Y_predict_TA=H_model_A.predict_proba(X_tr_A) # Predict value of train dataset
        Y_predict_VA=H_model_A.predict_proba(X_te_A)# Predict value of validation dataset
        Y_los_TA=round(sklearn.metrics.log_loss(y_train_A,Y_predict_TA),4)# calculate los for train sets
        Y_los_VA=round(sklearn.metrics.log_loss(y_test_A,Y_predict_VA),4)# calculate los for validation sets
        end_A=time.time()
        Input_row_A={"K": a, "train":Y_los_TA, "validation":Y_los_VA,'time':round((end_A-start_A),3)}#Collect input row
        H_Parameter_A=H_Parameter_A.append(Input_row_A,ignore_index=True)#Collect input information into table
        H_all_models_A.append(H_model_A)

    #Train and validate using Dataset B
    KF2=KFold(n_splits=N_splits,shuffle=True,random_state=19+i)   
    for train_index_B, test_index_B in KF2.split(I_Data_B):
        #Normalize data
        b=b+1
        print(f"Dataset_B Loop:{b}")
        y_train_B=I_Data_B.iloc[train_index_B]["Diabetes_binary"] # Preserve 0/1
        y_test_B=I_Data_B.iloc[test_index_B]["Diabetes_binary"]
        St_train_1B=StandardScaler()
        St_test_1B=StandardScaler()
        train_1B=pd.DataFrame(St_train_1B.fit_transform(I_Data_B.iloc[train_index_B]),columns=I_Data_B.columns)# Get train dataset
        test_1B=pd.DataFrame(St_test_1B.fit_transform(I_Data_B.iloc[test_index_B]),columns=I_Data_B.columns)   # Get test dataset
        X_tr_B=train_1B.drop(["Diabetes_binary"],axis=1) # Get X train
        X_te_B=test_1B.drop(["Diabetes_binary"],axis=1)   # Get X test     
        # Fit model A
        H_all_models_B = []
        #Start fit process
        start_B=time.time()
        H_model_B=sklearn.linear_model.LogisticRegression(dual=False,solver='sag',max_iter=8000) # Logistic regression
        H_model_B.fit(X_tr_B,y_train_B)# Train model
        Y_predict_TB=H_model_B.predict_proba(X_tr_B) # Predict value of train dataset
        Y_predict_VB=H_model_B.predict_proba(X_te_B)# Predict value of validation dataset
        Y_los_TB=round(sklearn.metrics.log_loss(y_train_B,Y_predict_TB),4)# calculate los for train sets
        Y_los_VB=round(sklearn.metrics.log_loss(y_test_B,Y_predict_VB),4)# calculate los for validation sets
        end_B=time.time()
        Input_row_B={"K": b, "train":Y_los_TB, "validation":Y_los_VB,'time':round((end_B-start_B),3)}#Collect input row
        H_Parameter_B=H_Parameter_B.append(Input_row_B,ignore_index=True)#Collect input information into table
        H_all_models_B.append(H_model_B)

#Print average error information for Dataset A & Dataset A
Train_mean_A=round(np.mean(H_Parameter_A["train"]),4)
Vali_mean_A=round(np.mean(H_Parameter_A["validation"]),4)
Time_A=round(np.mean(H_Parameter_A["time"]),4)
Train_mean_B=round(np.mean(H_Parameter_B["train"]),4)
Vali_mean_B=round(np.mean(H_Parameter_B["validation"]),4)
Time_B=round(np.mean(H_Parameter_B["time"]),4)
print(f"Train Los of Dataset A is {Train_mean_A},Validate Los of Dataset A is {Vali_mean_A}")
print(f"Average Time Cost of Dataset A is {Time_A}")
print("------------------------------------------------------------------------------------")
print(H_Parameter_A)
print("------------------------------------------------------------------------------------")
print(f"Train Los of Dataset B is {Train_mean_B},Validate Los of Dataset B is {Vali_mean_B}")
print(f"Average Time Cost of Dataset A is {Time_B}")
print("------------------------------------------------------------------------------------")
print(H_Parameter_B)

#Draw figure of hyper turning results
AA=plt.figure()
PIC1=AA.add_subplot(2,2,1)
sns.lineplot(x="K", y="train", data=H_Parameter_A, label="train")
sns.lineplot(x="K", y="validation", data=H_Parameter_A, label="validation")
plt.gca().set_ylabel("Los")
plt.title('Los of Dataset A')
PIC2=AA.add_subplot(2,2,2)
sns.lineplot(x="K", y="time", data=H_Parameter_A, label="train")
sns.lineplot(x="K", y="time", data=H_Parameter_A, label="validation")
plt.gca().set_ylabel("Time(s)")
plt.title('Time cost of Dataset B')
PIC1=AA.add_subplot(2,2,3)
sns.lineplot(x="K", y="train", data=H_Parameter_B, label="train")
sns.lineplot(x="K", y="validation", data=H_Parameter_B, label="validation")
plt.gca().set_ylabel("Los")
plt.title('Los of Dataset B')
PIC3=AA.add_subplot(2,2,4)
sns.lineplot(x="K", y="time", data=H_Parameter_B, label="train")
sns.lineplot(x="K", y="time", data=H_Parameter_B, label="validation")
plt.gca().set_ylabel("Time(s)")
plt.title('Time cost of Dataset B')
plt.tight_layout()
plt.show()
