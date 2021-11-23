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
N_splits=9   #Define K fold 
N_trails=1   #Define loops(random seed)
Order=1       #Define features order,0-oder not increase with K-fold,1-order increase with order
P_C=0.05         # Set penalty C start point
L1_Ratio_step=0.1   # Set penalty C increase step
R_solver='saga' # Define regression solver，newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
R_type_A=3      # Define regression type, 0-normal, 1-ridge, 2-lasso, 3-elastic net
Tol=1e-5      # Set tolerance
M_iter=22000   # Set max iter number
L1_ratio=0.1  # Set L1 ratio for elasticnet
H_Parameter_Kfold=pd.DataFrame(columns=["K","train","C", "validation",'time']) #Define output of Lasso
H_Parameter_Trail=pd.DataFrame(columns=["Type","C","Train_mean","Vali_mean","Time_Kfold","Time_Trail"]) #Define output of Ridge
#Split Data set to training and validations
#package function
def Regression(n_split:int,n_trail:int,order:int,p_c:float,l1_ratio_step:float,Itol:float,r_type:int,r_solver:str,m_iter:int,l1_ratio:float,data):
    I_data=pd.DataFrame(data,columns=data.columns)
    a=0            #Define trail counting number 
    I_solver=r_solver
    I_p_c=p_c
    I_m_iter=m_iter
    if  r_type==0 : #Define characters based on regression type
        I_penalty="none"
        I_p_c="none"      
    elif r_type==1 :
        I_penalty="l1"
        I_solver="saga"
    elif r_type==2 :
        I_penalty="l2"
    elif r_type==3 :
        I_penalty='elasticnet'
        
    H_parameter_kfold=pd.DataFrame(columns=["K", "train", "validation",'time',"L1 Ratio"])
    H_parameter_trail=pd.DataFrame(columns=["Type","L1 Ratio","Train_mean","Vali_mean","Time_Kfold","Time_Trail"])  
    for i in range(n_trail):
        b=0            #Define K loop counting number
        a=a+1
        I_l1_ratio=l1_ratio 
        #print(f"L1 Ratio:{E_ratio} Loop:{a}") # Count loop number
        start_trail=time.time()
        #Train and validate
        KF=KFold(n_splits=n_split,shuffle=True,random_state=19)   
        for train_index, test_index in KF.split(I_data):
            b=b+1
            print(f"L1 Ratio:{I_l1_ratio} in K_fold Loop:{b}")
            print(I_data.shape)
            y_train=I_data.iloc[train_index]["Diabetes_binary"] 
            y_test=I_data.iloc[test_index]["Diabetes_binary"]
            X_tr=I_data.iloc[train_index]
            X_te=I_data.iloc[test_index]
            X_tr=X_tr.drop(["Diabetes_binary"],axis=1)   # Get X train
            X_te=X_te.drop(["Diabetes_binary"],axis=1)   # Get X test     
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
            Input_row={"K": b, "train":Y_los_T, "L1 Ratio":I_l1_ratio,"validation":Y_los_V,'time':round((end_Kfold-start_Kfold),4)}#Collect input row
            H_parameter_kfold=H_parameter_kfold.append(Input_row,ignore_index=True)#Collect input information into table
            I_l1_ratio=I_l1_ratio+l1_ratio_step
            H_all_models.append(H_model)
            
                # Get trail results
        end_trail=time.time()
        Train_mean=round(np.mean(H_parameter_kfold["train"]),4)
        Vali_mean=round(np.mean(H_parameter_kfold["validation"]),4)
        Time_Kfold=round(np.mean(H_parameter_kfold["time"]),4)
        if r_type==0:
           r_type_I="Normal"
        elif r_type==1:
           r_type_I="Ridge"
        elif r_type==2:
           r_type_I="Lasso"
        elif r_type==3:
           r_type_I="ElasticNet"           
        Input_row_trail={"Type":r_type_I,"L1 Ratio":I_l1_ratio-l1_ratio_step,"Train_mean":Train_mean,"Vali_mean":Vali_mean,"Time_Kfold":Time_Kfold,"Time_Trail":round((end_trail-start_trail),4)}
        H_parameter_trail=H_parameter_trail.append(Input_row_trail,ignore_index=True)#Collect input information into table
    return H_parameter_kfold, H_parameter_trail

#main calc
H_Parameter_Kfold,H_Parameter_Trail=Regression(n_split=N_splits,n_trail=N_trails,order=Order,p_c=P_C,l1_ratio_step=L1_Ratio_step,Itol=Tol,r_type=R_type_A,
r_solver=R_solver,m_iter=M_iter,l1_ratio=L1_ratio,data=Data_B)      
#Print results
print("------------------Results------------------")
print(H_Parameter_Kfold)
print("-------------------------------------------")
print(H_Parameter_Trail)
print("-------------------------------------------")
#Print results
AA=plt.figure()
PIC1=AA.add_subplot(1,2,1)
sns.lineplot(x="L1 Ratio", y="train", data=H_Parameter_Kfold, label="train")
sns.lineplot(x="L1 Ratio", y="validation", data=H_Parameter_Kfold, label="validation")
plt.gca().set_ylabel("Los")
plt.title('Los for different L1 Ratio')
plt.tight_layout()
PIC1=AA.add_subplot(1,2,2)
sns.lineplot(x="L1 Ratio", y="time", data=H_Parameter_Kfold, label="Time Cost")
plt.gca().set_ylabel("Time(s)")
plt.title('Time for different L1 Ratio')
plt.tight_layout()

BB=plt.figure()
ax = plt.axes(projection='3d')
if Order==1:
    x=H_Parameter_Kfold[:N_trails]["K"]
else:
    x=H_Parameter_Kfold[:N_trails]["K"]
    for z in range(len(x)):
        x.iloc[z]=1 
y=H_Parameter_Trail[:N_trails]["L1 Ratio"]
z=pd.DataFrame(index=x.index,columns=y.index)
for p in range(len(x)):
    for v in range(len(y)):
        z.iloc[p,v]=H_Parameter_Kfold.loc[p*3+v,"validation"]
X, Y = np.meshgrid(x, y)
plt.xlabel("Order")
plt.ylabel("L1 Ratio")
ax.set_title('Los for different L1 Ratio')
ax.scatter3D(X, Y, z)
plt.tight_layout()
plt.show()