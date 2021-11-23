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
y_B=I_Data_B["Diabetes_binary"] # Preserve Y 0/1
x_BS=StandardScaler()#Normalize data
Data_B=pd.DataFrame(x_BS.fit_transform(I_Data_B.iloc[:,1:]),columns=I_Data_B.columns[1:])
Data_B["Diabetes_binary"]=y_B # Put Y Back
#Define Variables

N_trails=1   #Define loops(random seed)
P_C=0.05         # Set penalty C start point
R_solver='liblinear' # Define regression solver，newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
R_type_A=3      # Define regression type, 0-normal, 1-ridge, 2-lasso, 3-elastic net
Tol=1e-5      # Set tolerance
M_iter=22000   # Set max iter number

H_Parameter_Kfold=pd.DataFrame(columns=["K","train","C", "validation",'time']) #Define output of Lasso

#package function
def Regression(n_trail:int,p_c:float,Itol:float,r_type:int,r_solver:str,m_iter:int,data):
    I_data=pd.DataFrame(data,columns=data.columns)
    a=0            #Define trail counting number 
    I_solver=r_solver
    I_p_c=p_c
    I_penalty='l1'
    H_parameter_kfold=pd.DataFrame(columns=["K", "train", "validation",'time'])
    
    for i in range(n_trail):
        a=a+1            #Define K loop counting number
        #print(f"L1 Ratio:{E_ratio} Loop:{a}") # Count loop number
        start_trail=time.time()
        #Train and validate
        I_data_x=I_data.drop(["Diabetes_binary"],axis=1) # Get X
        I_data_y=I_data["Diabetes_binary"]
        X_tr, X_te, y_train, y_test = train_test_split(I_data_x, I_data_y, test_size=0.0005, random_state=(19))  
        #print(f"L1 Ratio:{I_l1_ratio} in K_fold Loop:{a}")
        #print(I_data.shape)
        # Fit model
        H_all_models = []
        #Start fit process
        start_Kfold=time.time()
        H_model=sklearn.linear_model.LogisticRegression(penalty=I_penalty,dual=False,solver=I_solver,
        tol=Itol,C=I_p_c,max_iter=m_iter)                          # Logistic regression
        H_model.fit(X_tr,y_train)# Train model
        Y_predict_T=H_model.predict_proba(X_tr) # Predict value of train dataset
        Y_predict_V=H_model.predict_proba(X_te)# Predict value of validation dataset
        Y_los_T=round(sklearn.metrics.log_loss(y_train,Y_predict_T),4)# calculate los for train sets
        Y_los_V=round(sklearn.metrics.log_loss(y_test,Y_predict_V),4)# calculate los for validation sets
  
        fpr, tpr, thresholds = metrics.roc_curve(y_test,Y_predict_V[:,1],pos_label=1)# Calculate ROC
        auc=metrics.auc(fpr, tpr)
        # bootstrap
        Y_Test=pd.DataFrame(y_test)
        np.random.seed(19)
        B_size = 100 # number of bootstrap
        Auc=[]
        Y_los_VB=[]
        for _ in tqdm(range(B_size)):
            # bootstrap the indices and build the bootstrap data
            idx = np.random.randint(low=0, high=len(X_te), size=len(X_te))
            X_bootstrap = X_te.iloc[idx]
            Y_bootstrap=Y_Test.iloc[idx]
            Y_predict_VB=H_model.predict_proba(X_bootstrap)
            fpr, tpr, thresholds = metrics.roc_curve(Y_bootstrap,Y_predict_VB[:,1],pos_label=1)# Calculate ROC
            auc=metrics.auc(fpr, tpr)
            Y_Los_VB=sklearn.metrics.log_loss(Y_bootstrap,Y_predict_VB)
            Auc.append(auc)
            Y_los_VB.append(Y_Los_VB)
        end_Kfold=time.time()
        Input_row={"K": a, "train":Y_los_T, "validation":Y_los_V,'time':round((end_Kfold-start_Kfold),4)}#Collect input row
        H_parameter_kfold=H_parameter_kfold.append(Input_row,ignore_index=True)#Collect input information into table
        H_all_models.append(H_model)   
        # Get trail results
        end_trail=time.time()
        
    return Auc,Y_los_VB

#main calc
AUC,Y_LOS_VB=Regression(n_trail=N_trails,p_c=P_C,Itol=Tol,r_type=R_type_A,
r_solver=R_solver,m_iter=M_iter,data=Data_B)      
#Print results
print("------------------Results------------------")
print(H_Parameter_Kfold)
print(f'Average Los is {round(np.mean(Y_LOS_VB),4)}')
print(f'Average AUC is {round(np.mean(AUC),4)}')
print("-------------------------------------------")