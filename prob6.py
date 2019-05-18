# HW1 Problem6

import csv                                                                      # 讀CSV檔案
import numpy as np                                                              # 矩陣, 隨機變數
import math                                                                     # 開根號
import matplotlib.pyplot as plt                                                 # 畫圖使用


## Parameter
total_data=np.zeros([891,7])                                                          # 共有891筆data,每筆7個參數
project=['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']                # 7個參數


learning_rate=0.001


## 讀取data
point=0                                                                         # 目前取到CSV檔案中的第幾列
with open('titanic.csv', newline='') as csvFile:
    rows = csv.DictReader(csvFile)
    for row in rows:
        for profect_number in range(0,7):
            total_data[point,profect_number]=row[project[profect_number]]
        point+=1

## 統計
       
###############################################################################    
# 與PClass的關聯

PClass1_survived=0
PClass2_survived=0
PClass3_survived=0
PClass1_num=0
PClass2_num=0
PClass3_num=0

for data_pointer in range(0,np.size(total_data,0)):
    if total_data[data_pointer,1]==1:      # 屬於PClass1
        PClass1_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            PClass1_survived+=1
    elif total_data[data_pointer,1]==2:      # 屬於PClass2
        PClass2_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            PClass2_survived+=1
    else:
        PClass3_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            PClass3_survived+=1

PClass=[PClass1_survived/PClass1_num,PClass2_survived/PClass2_num,PClass3_survived/PClass3_num]
plt.figure(1)
plt.plot(PClass)
plt.xlabel('PClass')
plt.ylabel('Survived rate')
plt.show() 



###############################################################################
# 與Sex的關聯

Sex0_survived=0
Sex1_survived=0
Sex0_num=0
Sex1_num=0

for data_pointer in range(0,np.size(total_data,0)):
    if total_data[data_pointer,2]==0:           # 屬於Sex0
        Sex0_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Sex0_survived+=1
    else:
        Sex1_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Sex1_survived+=1

Sex=[Sex0_survived/Sex0_num,Sex1_survived/Sex1_num]
plt.figure(2)
plt.plot(Sex)
plt.xlabel('Sex')
plt.ylabel('Survived rate')
plt.show() 

###############################################################################
# 與Age的關聯

Age1_survived=0
Age2_survived=0
Age3_survived=0
Age1_num=0
Age2_num=0
Age3_num=0


for data_pointer in range(0,np.size(total_data,0)):
    if total_data[data_pointer,3]<25:                  # 屬於Age1
        Age1_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Age1_survived+=1
    elif total_data[data_pointer,3]<50:      # 屬於Age2
        Age2_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Age2_survived+=1
    else:
        Age3_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Age3_survived+=1

Age=[Age1_survived/Age1_num,Age2_survived/Age2_num,Age3_survived/Age3_num]
plt.figure(3)
plt.plot(Age)
plt.xlabel('Age Class')
plt.ylabel('Survived rate')
plt.show() 

      

###############################################################################
# 與SibSp的關聯

SibSp0_survived=0
SibSp1_survived=0
SibSp0_num=0
SibSp1_num=0


for data_pointer in range(0,np.size(total_data,0)):
    if total_data[data_pointer,4]==0:                  # 沒有SibSp
        SibSp0_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            SibSp0_survived+=1
    else:
        SibSp1_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            SibSp1_survived+=1

SibSp=[SibSp0_survived/SibSp0_num,SibSp1_survived/SibSp1_num]
plt.figure(4)
plt.plot(SibSp)
plt.xlabel('SibSp existed')
plt.ylabel('Survived rate')
plt.show() 

###############################################################################
# 與Parch的關聯

Parch0_survived=0
Parch1_survived=0
Parch0_num=0
Parch1_num=0


for data_pointer in range(0,np.size(total_data,0)):
    if total_data[data_pointer,5]==0:                  # 沒有Parch
        Parch0_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Parch0_survived+=1
    else:
        Parch1_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Parch1_survived+=1

Parch=[Parch0_survived/Parch0_num,Parch1_survived/Parch1_num]
plt.figure(5)
plt.plot(Parch)
plt.xlabel('Parch existed')
plt.ylabel('Survived rate')
plt.show() 



###############################################################################
# 與Fare的關聯

Fare1_survived=0
Fare2_survived=0
Fare3_survived=0
Fare1_num=0
Fare2_num=0
Fare3_num=0


for data_pointer in range(0,np.size(total_data,0)):
    if total_data[data_pointer,6]<50:                  # 屬於Fare1
        Fare1_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Fare1_survived+=1
    elif total_data[data_pointer,6]<100:      # 屬於Fare2
        Fare2_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Fare2_survived+=1
    else:
        Fare3_num+=1
        if total_data[data_pointer,0]==1:       # 若此人最後存活
            Fare3_survived+=1

Fare=[Fare1_survived/Fare1_num,Fare2_survived/Fare2_num,Fare3_survived/Fare3_num]
plt.figure(6)
plt.plot(Fare)
plt.xlabel('Fare Class')
plt.ylabel('Survived rate')
plt.show() 

      


###############################################################################




data=total_data[:800]                                                           # For training data
data_test=total_data[800:]                                                      # For testing data
############################  Function  #######################################        
        
       
# 計算此次data丟入後，每層的輸出結果與output y
def Forward_DNN(w2,w3,w4,b2,b3,b4,x1):
    x2=np.dot(x1,w2)+b2   
    x3=np.dot(x2,w3)+b3    
    y=np.dot(x3,w4)+b4
    return x2 ,x3 ,y


def Backward_DNN(w2,w3,w4,x1,x2,x3,y,survived):
    # survived 為該人最後是否存活
    
    
    # (非通用))
    partial_E_partial_y=np.zeros([1,2])                                         # 已知 dE/dy
    
    if survived==1:
        partial_E_partial_y[0,:]=[y[0]-1,y[1]]                                    # [t*ln(y1) t*ln(y2)]
    else:
        partial_E_partial_y[0,:]=[y[0],y[1]-1]                                    # [t*ln(y1) t*ln(y2)]
    
    

    
    partial_E_partial_w4= np.dot( np.transpose(x3),partial_E_partial_y  )       # dE/dw4
    partial_E_partial_x3= np.dot( partial_E_partial_y,np.transpose(w4)  )       # dE/dx3
    partial_E_partial_w3= np.dot( np.transpose(x2),partial_E_partial_x3 )       # dE/dw3
    partial_E_partial_x2= np.dot( partial_E_partial_x3,np.transpose(w3) )       # dE/dx2
    partial_E_partial_w2= np.dot( np.transpose(x1),partial_E_partial_x2 )       # dE/dw2
    
    partial_E_partial_b4= np.dot( partial_E_partial_y,np.eye(np.size(partial_E_partial_w4,1))  )  # dE/db4
    partial_E_partial_b3= np.dot( partial_E_partial_x3,np.eye(np.size(partial_E_partial_w3,1))  ) # dE/db3
    partial_E_partial_b2= np.dot( partial_E_partial_x2,np.eye(np.size(partial_E_partial_w2,1))  ) # dE/db2
    
    return partial_E_partial_w2,partial_E_partial_w3,partial_E_partial_w4,partial_E_partial_b2,partial_E_partial_b3,partial_E_partial_b4
                     
 

############################### Main  #########################################

## Mini-batch SGD
    
# 設置節點數量
x1_length=6
x2_length=3
x3_length=3
y_length=2    

## Batch參數
batch_size=10                                                                   # 每個batch有10筆資料
batch_num=int(800/batch_size)                                                   # 總共分成多少個batch

# 初始化Weight網路
w2=np.random.uniform(-1/math.sqrt(6),1/math.sqrt(6),(x1_length,x2_length))      # 隨機生成uniform的6*3矩陣
w3=np.random.uniform(-1/math.sqrt(3),1/math.sqrt(3),(x2_length,x3_length))      # 隨機生成uniform的3*3矩陣
w4=np.random.uniform(-1/math.sqrt(3),1/math.sqrt(3),(x3_length,y_length))       # 隨機生成uniform的3*2矩陣 

# 初始化Bias網路
b2=np.random.uniform(-1/math.sqrt(3),1/math.sqrt(3),(1,x2_length))              # 隨機生成uniform的1*3矩陣
b3=np.random.uniform(-1/math.sqrt(3),1/math.sqrt(3),(1,x3_length))              # 隨機生成uniform的1*3矩陣
b4=np.random.uniform(-1/math.sqrt(2),1/math.sqrt(2),(1,y_length))               # 隨機生成uniform的1*2矩陣 


# 暫存節點數值
x1_temp=np.zeros([batch_size,x1_length])                                        # 暫存這個batch的所有x1
x2_temp=np.zeros([batch_size,x2_length])                                        # 暫存這個batch的所有x2
x3_temp=np.zeros([batch_size,x3_length])                                        # 暫存這個batch的所有x3
y_temp=np.zeros([batch_size,y_length])                                          # 暫存這個batch的所有y

# 紀錄數值
training_loss=[]                                                                # 每次epoch的E值
training_error_rate=[]                                                          # training data的error
testing_error_rate=[]                                                           # testing data的error


for epoch in range(0,1000):
    
    
    # 把全部training data分成80等分,每等分有10個data
    
    total_cross_entropy=0                                                       # 這個batch的所有E累加

    for batch_number in range(0,batch_num):
        
        

        
        E=0                                                                             # Error

        for data_number in range(0,batch_size):
            x1_temp[data_number,:]=data[batch_number*batch_size+data_number,1:]                 # 存入輸入的第一層data
            # 代入Forward DNN計算每層的輸出
            x2_temp[data_number,:],x3_temp[data_number,:],y_temp[data_number,:]=Forward_DNN(w2,w3,w4,b2,b3,b4,x1_temp[data_number,:])
        
        
            # 把output做soft max
            exp_sum=0 
            for i in range(0,y_length):                                                 # 先計算exp總和
                exp_sum +=math.exp(y_temp[data_number,i])
            for i in range(0,y_length):                                                 # 計算經過softmax的結果
                y_temp[data_number,i]=math.exp(y_temp[data_number,i])/exp_sum
        
        
        # 計算error
        for data_number in range(0,batch_size):
    
            if data[batch_number*batch_size+data_number,0]==1:                                  # 這個人最後會存活

                E-=  math.log(  y_temp[data_number,0]  )
        
            else:
                E-=  math.log(  y_temp[data_number,1]  )

        total_cross_entropy+=E                                                  # Cross entropy累加                                               
        
        # SGD
        
        total_partial_E_partial_w2=np.zeros([x1_length,x2_length])  
        total_partial_E_partial_w3=np.zeros([x2_length,x3_length])
        total_partial_E_partial_w4=np.zeros([x3_length,y_length])
        
        total_partial_E_partial_b2=np.zeros([1,x2_length]) 
        total_partial_E_partial_b3=np.zeros([1,x3_length]) 
        total_partial_E_partial_b4=np.zeros([1,y_length]) 
        
        
  
        
        
        for data_number in range(0,batch_size):                                         # 依序將每個結果去算backward求梯度
    
            partial_E_partial_w2,partial_E_partial_w3,partial_E_partial_w4,partial_E_partial_b2,partial_E_partial_b3,partial_E_partial_b4= Backward_DNN(w2,w3,w4,[x1_temp[data_number,:]],[x2_temp[data_number,:]],[x3_temp[data_number,:]],y_temp[data_number,:],data[batch_number*batch_size+data_number,0])
            total_partial_E_partial_w2+=partial_E_partial_w2                            # partial_E_partial_w2 梯度累加
            total_partial_E_partial_w3+=partial_E_partial_w3                            # partial_E_partial_w3 梯度累加
            total_partial_E_partial_w4+=partial_E_partial_w4                            # partial_E_partial_w4 梯度累加
            
            total_partial_E_partial_b2+=partial_E_partial_b2                            # partial_E_partial_b2 梯度累加
            total_partial_E_partial_b3+=partial_E_partial_b3                            # partial_E_partial_b3 梯度累加
            total_partial_E_partial_b4+=partial_E_partial_b4                            # partial_E_partial_b4 梯度累加
            
            
    
        # 更新w1,w2,w3

        # 先把累計梯度做平均
        
        average_partial_E_partial_w2=total_partial_E_partial_w2/batch_size
        average_partial_E_partial_w3=total_partial_E_partial_w3/batch_size
        average_partial_E_partial_w4=total_partial_E_partial_w4/batch_size
        
        average_partial_E_partial_b2=total_partial_E_partial_b2/batch_size
        average_partial_E_partial_b3=total_partial_E_partial_b3/batch_size
        average_partial_E_partial_b4=total_partial_E_partial_b4/batch_size
        
        

        # 更新梯度
        
        w2=w2-learning_rate *average_partial_E_partial_w2
        w3=w3-learning_rate *average_partial_E_partial_w3
        w4=w4-learning_rate *average_partial_E_partial_w4
        
        b2=b2-learning_rate *average_partial_E_partial_b2
        b3=b3-learning_rate *average_partial_E_partial_b3
        b4=b4-learning_rate *average_partial_E_partial_b4
    
    # 計算這次epoch的平均cross entropy
    training_loss.append(total_cross_entropy/batch_num)
    
    
    # 計算training error rate
    accuracy=0                                                                  # 判斷正確的資料數
    for train_pointer in range(0,np.size(data,0)):                              # 每筆traing data
        x1_train=data[train_pointer,1:]                                         # 取得某筆train data的input
        x2_unused,x3_unused,y_train=Forward_DNN(w2,w3,w4,b2,b3,b4,x1_train)     # 得到train data的output
        
        # 把output做soft max
        exp_sum=0 
        for i in range(0,np.size(y_train,1)):                                   # 先計算exp總和
            exp_sum +=math.exp(y_train[0,i])
        for i in range(0,np.size(y_train,1)):                                    # 計算經過softmax的結果
            y_train[0,i]=math.exp(y_train[0,i])/exp_sum
 
        
        if data[train_pointer,0]==1 and y_train[0,0]>0.5:
            accuracy+=1
        elif data[train_pointer,0]==0 and y_train[0,1]>0.5:
            accuracy+=1
    error_rate=1- (accuracy/np.size(data,0))
    training_error_rate.append(error_rate)
    
    
    # 計算testing error rate
    accuracy=0                                                                  # 判斷正確的資料數
    for test_pointer in range(0,np.size(data_test,0)):                            # 每筆testing data
        x1_test=data_test[test_pointer,1:]                                      # 取得某筆test data的input
        x2_unused,x3_unused,y_test=Forward_DNN(w2,w3,w4,b2,b3,b4,x1_test)                # 得到test data的output
        
        # 把output做soft max
        exp_sum=0 
        for i in range(0,np.size(y_test,1)):                                    # 先計算exp總和
            exp_sum +=math.exp(y_test[0,i])
        for i in range(0,np.size(y_test,1)):                                    # 計算經過softmax的結果
            y_test[0,i]=math.exp(y_test[0,i])/exp_sum
 
        
        if data_test[test_pointer,0]==1 and y_test[0,0]>0.5:
            accuracy+=1
        elif data_test[test_pointer,0]==0 and y_test[0,1]>0.5:
            accuracy+=1
    error_rate=1- (accuracy/np.size(data_test,0))
    print("Error rate=",error_rate)
    testing_error_rate.append(error_rate)
    
    
    
    # Shuffle
    np.random.shuffle(data)                                                     # 把training data順序打亂
    

###############################代入 articially design###########################

x_survived=np.zeros([1,6])                                                      # 預計會存活的data



x_survived=[1,0,30,2,2,200]  #Pclass=1,Sex=0,Age=30,SibSp=2,Parch=2,Fare=200

'''
x_survived=[3,1,70,0,0,10]  #Pclass=3,Sex=1,Age=70,SibSp=0,Parch=0,Fare=10
'''

x2_unused,x3_unused,y_survived=Forward_DNN(w2,w3,w4,b2,b3,b4,x_survived)            # 得到x_survived的output
        
# 把output做soft max
exp_sum=0 
for i in range(0,np.size(y_survived,1)):                                    # 先計算exp總和
    exp_sum +=math.exp(y_survived[0,i])
for i in range(0,np.size(y_survived,1)):                                    # 計算經過softmax的結果
    y_survived[0,i]=math.exp(y_survived[0,i])/exp_sum
    
# Result
print("The guy survived probobility=",y_survived[0,0])
     

    






 
        
        
