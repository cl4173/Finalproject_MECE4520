import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义图像和三维格式坐标轴
fig = plt.figure()
ax1 = Axes3D(fig)

z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
print(zd.shape)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
print(xd.shape)
ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()
