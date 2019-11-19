# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:48:57 2019

@author: water
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

import pickle
import os
import matplotlib.pyplot as plt

Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
filenamelist = []
Ball_x_position=[]
Ball_y_position=[]
PlatformPosition_x_position=[]
log = 'C:\\ML\\MLGame-master1118\\log'
filenames = os.listdir(log)

for filename in filenames:
    fileroutine = log + '\\' + filename
    with open(fileroutine,"rb") as f:
        data_list = pickle.load(f)
    for i in range(0,len(data_list)):
        Frame.append(data_list[i].frame)
        Status.append(data_list[i].status)
        Ballposition.append(data_list[i].ball)
        Ball_x_position.append(Ballposition[i][0])
        Ball_y_position.append(Ballposition[i][1])
        PlatformPosition.append(data_list[i].platform)
        PlatformPosition_x_position.append(PlatformPosition[i][0])
        Bricks.append(data_list[i].bricks)
"""
#创建图并命名
plt.figure('All Pickle file')
ax = plt.gca()
#设置x轴、y轴名称
ax.set_xlabel('Ball_x_position')
ax.set_ylabel('Ball_y_position')

#画散点图，以x_list中的值为横坐标，以y_list中的值为纵坐标
#参数c指定点的颜色，s指定点的大小,alpha指定点的透明度
ax.scatter(Ball_x_position, Ball_y_position, c='b', s=10, alpha=0.5)
plt.show()

#创建图并命名
plt.figure('All Pickle file')
ax = plt.gca()
#设置x轴、y轴名称
ax.set_xlabel('Frame')
ax.set_ylabel('PlatformPosition_x_position')

#画散点图，以x_list中的值为横坐标，以y_list中的值为纵坐标
#参数c指定点的颜色，s指定点的大小,alpha指定点的透明度
ax.scatter(Frame, PlatformPosition_x_position, c='b', s=10, alpha=0.5)
plt.show()
"""
"""plt.scatter(Frame, PlatformPosition_x_position)
plt.show()"""

import numpy as np
print(PlatformPosition)
print("\n\n\n\n")
PlatX=np.array(PlatformPosition)[:,0][:,np.newaxis] #[:,0] -> 全部的X取出 [:,np.newaxis] -> 建立新的維度為 len(PlatformPosition) x 1
print(PlatX)
print("\n\n\n\n")
PlatX_next=PlatX[1:,:]  #[1:,:] =>  [1:, ->從陣列第一點開始到全部   ,: ->剩餘的每個維度裡面所有的內容(y)
print(PlatX_next)
print("\n\n\n\n")
instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5    #全部的下一點平台 減去 平台的目前狀態只取X軸，每個數值建立一個新的維度，且每個點除以5，因為球的大小為5，移動輛差距也為5必須除回來
print(instruct)
print("\n\n\n\n")

Ballarray=np.array(Ballposition[:-1])
print(Ballarray)
print("\n\n\n\n")
ball_next=np.array(Ballposition[1:])
print(ball_next)
print("\n\n\n\n")

x=np.hstack((Ballarray,ball_next,PlatX[0:-1,0][:,np.newaxis]))  #將球X/球Y/球下一點X/球下一點Y/平台的X 每一筆建立一個新的維度
print(x)
print("\n\n\n\n")

y=instruct
print(y)

#x => 特徵值
#y => 該點特徵最終結果


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0) #20 0.928571  15 0.942857 25 0.957143 30 0.942857  28 0.857143

print(x_train)
print("\n\n\n\n")
print(x_test)
print("\n\n\n\n")
print(y_train)
print("\n\n\n\n")
print(y_test)
print("\n\n\n\n")
##
from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score
 

svm=SVC(C=1,gamma=0.009,class_weight='balanced',decision_function_shape='ovo') #kernel='linear',class_weight='balanced', gamma='auto'    #C為懲罰係數，C越高，越不容許誤差，容易過擬合；C越小，對於誤差允許較為寬鬆，容易欠擬合
#svm = SVR(gamma=0.001,C = 1,epsilon = 0.1,kernel = 'rbf')
#svm.fit(x_train, y_train) #svm.fit(x_train, y_train.ravel())
svm.fit(x_train, y_train)

#print(x_test)
y_knn=svm.predict(x_test)
acc=accuracy_score(y_knn, y_test)
print(acc)
#from sklearn.metrics import r2_score#R square

#R2 = r2_score(y_test,y_knn)
#print('R2 = ',R2)
#print(len(Frame))
"""from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)

x_train_stdnorm=scaler.transform(x_train)

neigh.fit(x_train_stdnorm, y_train)

x_test_stdnorm=scaler.transform(x_test)
yt=neigh.predict(x_test_stdnorm)

acc=accuracy_score(yt, y_test)"""



filename="SVC_example.sav"
pickle.dump(svm, open(filename, 'wb'))


l_model=pickle.load(open(filename, 'rb'))
yp_l=l_model.predict(x_test)
print("acc load: %f " % accuracy_score(yp_l, y_test))