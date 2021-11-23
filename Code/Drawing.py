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

H_Parameter_Kfold=pd.DataFrame(data=[[1.0,0.3196,0.3208,1.8417,0.05],[2.0,0.3147,0.3136,1.7935,0.05],[3.0,0.3150,0.3077,18.9474,0.05],[4.0,0.3145,0.3123,21.3786,0.05],
[5.0,0.3144,0.3128,35.7376,0.05],[6.0,0.3138,0.3196,188.5810,0.05],[7.0,0.3144,0.3131,239.3796,0.05],[8.0,0.3143,0.3137,368.2776,0.05],[9.0,0.3144,0.3127,601.0902,0.05],
[10.0,0.3139,0.3177,441.3373,0.05],[11.0,0.3138,0.3197,770.9004,0.05],[12.0,0.3142,0.3148,386.5137,0.05]],columns=["K","train", "validation",'time',"C"],
index=[0,1,2,3,4,5,6,7,8,9,10,11]) #Define output of Lasso

print(H_Parameter_Kfold)
#Print results
AA=plt.figure()
PIC1=AA.add_subplot(1,2,1)
sns.lineplot(x="K", y="train", data=H_Parameter_Kfold, label="train")
sns.lineplot(x="K", y="validation", data=H_Parameter_Kfold, label="validation")
plt.gca().set_ylabel("Los")
plt.title('Los with different order')
PIC1=AA.add_subplot(1,2,2)
sns.lineplot(x="K", y="time", data=H_Parameter_Kfold, label="Time Cost")
plt.gca().set_ylabel("Time(s)")
plt.title('time with different order')
plt.show()
plt.imshow(AA)
