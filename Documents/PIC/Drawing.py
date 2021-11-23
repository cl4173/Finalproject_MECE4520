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

H_Parameter_Kfold=pd.DataFrame(data=[[1.0,0.3144,0.3143,44.8799,0.1],[2.0,0.3144,0.3144,50.3343,0.2],[3.0,0.3144,0.3144,65.6750,0.3],[4.0,0.3144,0.3144,63.0800,0.4],
[5.0,0.3144,0.3144,73.4274,0.5],[6.0,0.3144,0.3144,88.7764,0.6],[7.0,0.3144,0.3144,116.9405,0.7],[8.0,0.3144,0.3144,169.0301,0.8],[9.0,0.3144,0.3144,94.3548,0.9]],
columns=["K","train", "validation",'time',"L1 Ratio"],index=[0,1,2,3,4,5,6,7,8]) #Define output of Lasso

print(H_Parameter_Kfold)
#Print results
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
plt.show()