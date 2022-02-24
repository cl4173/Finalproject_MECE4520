# Import libs
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from pandas.tseries.offsets import BYearBegin
import seaborn as sns
import time
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes.base import Index, ensure_index
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from tqdm import tqdm
import joblib
import warnings
#warnings.filterwarnings("ignore")

# Import data_B & clean
URL=r'C:\Users\liuch\Desktop\Courses\Data_Science_MECE4520\Project\diabetes_binary_health_indicators_BRFSS2015.csv'
I_Data_B=pd.read_csv(URL)
I_Data_B=pd.DataFrame(I_Data_B,columns=I_Data_B.columns)
I_Data_B.fillna(0)
for b in range(2,5):   # Add high orders components (5)         
    I_Data_B[f"BMI{b}"] =I_Data_B["BMI"].apply(lambda x: x**(b))
    I_Data_B[f"GenHlth{b}"] =I_Data_B["GenHlth"].apply(lambda x: x**(b))
    I_Data_B[f"MentHlth{b}"] =I_Data_B["MentHlth"].apply(lambda x: x**(b))
    I_Data_B[f"PhysHlth{b}"] = I_Data_B["PhysHlth"].apply(lambda x: x**(b))
    I_Data_B[f"Age{b}" ] = I_Data_B["Age"].apply(lambda x: x**(b))
    I_Data_B[f"Education{b}" ] = I_Data_B["Education"].apply(lambda x: x**(b))
    I_Data_B[f"Income{b}"] = I_Data_B["Income"].apply(lambda x: x**(b))
Data_B=pd.DataFrame(I_Data_B,columns=I_Data_B.columns)
#Define Variables

N_trails=1   #Define loops(random seed)
P_C=0.05         # Set penalty C start point
R_solver='saga' # Define regression solver，newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
R_type_A=3      # Define regression type, 0-normal, 1-ridge, 2-lasso, 3-elastic net
Tol=1e-5      # Set tolerance
M_iter=22000   # Set max iter number
L1_ratio=0.5  # Set L1 ratio for elasticnet
H_Parameter_Kfold=pd.DataFrame(columns=["K","train","C", "validation",'time']) #Define output of Lasso

#package function
def Regression(n_trail:int,p_c:float,Itol:float,r_type:int,r_solver:str,m_iter:int,l1_ratio:float,data):
    I_data=pd.DataFrame(data,columns=data.columns)
    a=0            #Define trail counting number 
    I_solver=r_solver
    I_p_c=p_c
    I_penalty='elasticnet'
    H_parameter_kfold=pd.DataFrame(columns=["K", "train", "validation",'time',"L1 Ratio"])
    H_parameter_trail=pd.DataFrame(columns=["Type","L1 Ratio","Train_mean","Vali_mean","Time_Kfold","Time_Trail"])  
    I_l1_ratio=l1_ratio 
    for i in range(n_trail):
        a=a+1            #Define K loop counting number
        #print(f"L1 Ratio:{E_ratio} Loop:{a}") # Count loop number
        start_trail=time.time()
        #Train and validate
        I_data_x=I_data.drop(["Diabetes_binary"],axis=1) # Get X
        I_data_y=I_data["Diabetes_binary"]
        x_tr, x_te, y_train, y_test = train_test_split(I_data_x, I_data_y, test_size=0.0005, random_state=(19))  
        x_BS=StandardScaler()#Normalize data
        X_tr=x_BS.fit_transform(x_tr)
        X_te=x_BS.transform(x_te)
            
        #print(f"L1 Ratio:{I_l1_ratio} in K_fold Loop:{a}")
        #print(I_data.shape)
        # Fit model
        H_all_models = []
        #Start fit process
        
        start_Kfold=time.time()
        H_model=sklearn.linear_model.LogisticRegression(penalty=I_penalty,dual=False,solver=I_solver,
        tol=Itol,C=I_p_c,max_iter=m_iter,l1_ratio=I_l1_ratio,n_jobs=-1)                          # Logistic regression
        H_model.fit(X_tr,y_train)# Train model
        Y_predict_T=H_model.predict_proba(X_tr) # Predict value of train dataset
        Y_predict_V=H_model.predict_proba(X_te)# Predict value of validation dataset
        Y_los_T=round(sklearn.metrics.log_loss(y_train,Y_predict_T),4)# calculate los for train sets
        Y_los_V=round(sklearn.metrics.log_loss(y_test,Y_predict_V),4)# calculate los for validation sets
  
        end_Kfold=time.time()
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test,Y_predict_V[:,1],pos_label=1)# Calculate ROC
        auc=metrics.auc(fpr, tpr)
           
        end_Kfold=time.time()
        joblib.dump(H_model,r"C:\Users\liuch\Desktop\Logistic_Regression.m")   
        Input_row={"K": a, "train":Y_los_T, "L1 Ratio":I_l1_ratio,"validation":Y_los_V,'time':round((end_Kfold-start_Kfold),4)}#Collect input row
        H_parameter_kfold=H_parameter_kfold.append(Input_row,ignore_index=True)#Collect input information into table
        H_all_models.append(H_model)   
        # Get trail results
        end_trail=time.time()
        
    return H_parameter_kfold,auc,fpr,tpr,thresholds

#main calc
H_Parameter_Kfold,AUC,FPR,TPR,Thresholds=Regression(n_trail=N_trails,p_c=P_C,Itol=Tol,r_type=R_type_A,
r_solver=R_solver,m_iter=M_iter,l1_ratio=L1_ratio,data=Data_B)      
#Print results
print("------------------Results------------------")
print(H_Parameter_Kfold)
print(f'AUC is {round(AUC,4)}')
print("-------------------------------------------")

#Print results
AA=plt.figure()
sns.lineplot(x=FPR, y=TPR,  label="ROC")
sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--")
plt.gca().set_ylabel("TPR")
plt.gca().set_xlabel("FPR")
plt.title('ROC of ElasticNet Regression')
plt.tight_layout()

plt.show()
