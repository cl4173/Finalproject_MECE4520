import joblib
import numpy as np
from numpy.lib.npyio import zipfile_factory
import pandas as pd
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from tqdm import tqdm
import seaborn as sns
import sklearn
import time
data = pd.read_csv(r'C:\Users\liuch\Desktop\Courses\Data_Science_MECE4520\Project\diabetes_binary_health_indicators_BRFSS2015.csv')
data=pd.DataFrame(data)
# divide data into attributes and labels
X = data.drop(["Diabetes_binary"],axis=1)
y = data["Diabetes_binary"]
for j in range(0,2):
    X[f"BMI{j+2}"] =X["BMI"].apply(lambda x: x**(j+2))
    X[f"GenHlth{j+2}"] =X["GenHlth"].apply(lambda x: x**(j+2))
    X[f"MentHlth{j+2}"] = X["MentHlth"].apply(lambda x: x**(j+2))
    X[f"PhysHlth{j+2}"] = X["PhysHlth"].apply(lambda x: x**(j+2))
    X[f"Age{j+2}"] = X["Age"].apply(lambda x: x**(j+2))
    X[f"Education{j+2}"] = X["Education"].apply(lambda x: x**(j+2))
    X[f"Income{j+2}"] = X["Income"].apply(lambda x: x**(j+2))
# Divide into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0005, random_state = 42)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = pd.DataFrame(sc.transform(X_test))
y_test=pd.DataFrame(y_test)
print(X_train.shape)
print(X_test.shape)
#Define default characters,please keep
#Oob_score=True
Bootstrap=True
N_jobs=-1
Ccp_alpha=0
Max_leaf_nodes=None
''' Define variables, please pick no more than two variables 
------------------------------------------------------------'''

Criterion='gini'       # {“gini”, “entropy”}
Max_features='sqrt'    #{“auto”, “sqrt”, “log2”}
N_estimators=600      # set Tree number(start point),int
Max_depth=100          # set Max depth start point,int
Min_samples_leaf=76     # set Min depth start point,int
Min_samples_split=120    # set Min sample split start point,int
'''-------------------end of modify--------------------------'''

#Define output
Parameter=pd.DataFrame(columns=['Oob_score',"AUC","Los","Time"])
Roc=pd.DataFrame(columns=["fpr","tpr"])
Start = time.time()
regressor = RandomForestClassifier(criterion="gini", n_estimators = N_estimators,max_features=Max_features,max_depth=Max_depth,
max_leaf_nodes=Max_leaf_nodes,min_samples_leaf=Min_samples_leaf,min_samples_split=Min_samples_split,
bootstrap=Bootstrap,n_jobs=N_jobs,ccp_alpha=Ccp_alpha,random_state = 19)
regressor.fit(X_train, y_train)
Oob_Score=regressor.oob_score
y_pred = regressor.predict_proba(X_test)
End=time.time()
#calculate result
Los=round(sklearn.metrics.log_loss(y_test,y_pred),4)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred[:,1],pos_label=1)
AUC=metrics.auc(fpr, tpr)
np.random.seed(19)
B_size = 100 # number of bootstrap
Auc=[]
Y_los_VB=[]
for _ in tqdm(range(B_size)):
 # bootstrap the indices and build the bootstrap data
    idx = np.random.randint(low=0, high=len(X_test), size=len(X_test))
    X_bootstrap = X_test.iloc[idx]
    Y_bootstrap=y_test.iloc[idx]
    Y_predict_VB=regressor.predict_proba(X_bootstrap)
    fpr, tpr, thresholds = metrics.roc_curve(Y_bootstrap,Y_predict_VB[:,1],pos_label=1)# Calculate ROC
    auc=metrics.auc(fpr, tpr)
    Y_Los_VB=sklearn.metrics.log_loss(Y_bootstrap,Y_predict_VB)
    Auc.append(auc)
    Y_los_VB.append(Y_Los_VB)
#output
Input_row_P={'Oob_score':Oob_Score,"AUC":round(AUC,4),"Los":Los,"Time":round((End-Start),4)}
Input_row_R={"fpr":fpr,"tpr":tpr}
Parameter=Parameter.append(Input_row_P,ignore_index=True)
Roc=Roc.append(Input_row_R,ignore_index=True)
joblib.dump(regressor,r"C:\Users\liuch\Desktop\Randomforest.m")       

#Print results
print("------------------Results------------------")
print(Parameter)
print(f'Average Los is {round(np.mean(Y_Los_VB),4)}')
print(f'Average AUC is {round(np.mean(Auc),4)}')
print("-------------------------------------------")
#draw pictures
AA=plt.figure()
sns.lineplot(x=fpr, y=tpr,  label="ROC")
sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--")
plt.gca().set_ylabel("TPR")
plt.gca().set_xlabel("FPR")
plt.title('ROC of RandomForest')
plt.tight_layout()
plt.tight_layout()
plt.show()


#save data
#S_URL=(r'C:\Users\liuch\Desktop\R\{}.csv'.format(str(V1_name)))
#Parameter.to_csv(S_URL)