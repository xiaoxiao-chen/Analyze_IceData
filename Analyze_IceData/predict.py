#利用线性回归进行预测

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score,roc_curve,auc
from matplotlib import pyplot as plt

#-----读取已经归一化的数据--------
Normalized_data = pd.read_csv("train/21/21_Normalized_Data.csv")

#----将数据分成标签和属性列表-----
xList = []
names = []
labels = []

names1 = Normalized_data.columns.values.tolist()
names  = names1[:-1]
xList  = Normalized_data.iloc[:,:-1].values.tolist()
labels = Normalized_data.iloc[:,-1].values.tolist()

nrow = len(xList)       #179568行
ncol = len(xList[0])    #26列

#-------------------------------------------（1）线性回归预测---------------------------------------------
#------数据分为训练集、测试集，预留三分之一的数据------
indices = range(len(xList))
xListTest  = [xList[i] for i in indices if i%3 == 0 ]
xListTrain = [xList[i] for i in indices if i%3 != 0 ]
labelsTest  = [labels[i] for i in indices if i%3 == 0 ]
labelsTrain = [labels[i] for i in indices if i%3 != 0 ]

xTrain = np.array(xListTrain)       #(119712, 26)
xTest  = np.array(xListTest)        #(59856, 26)
yTrain = np.array(labelsTrain)      #(119712,)
yTest  = np.array(labelsTest)       #(59856,)

#------训练线性回归模型---------------------------
LinearRegressionModel = linear_model.LinearRegression()
LinearRegressionModel.fit(xTrain,yTrain)

#------对样本外数据进行预测--------------------------
testPredictions = LinearRegressionModel.predict(xTest)
#阈值设定？？？
#Ice = []
#indices = len(testPredictions)
#for i in range(indices):
#    if(testPredictions[i]>0.0):
#        ice = 1
#    else:
#        ice = 0
#    Ice.append(ice)
       
#画出预测值与实际值对比图
plt.figure()
plt.plot(range(len(testPredictions)),testPredictions,'b',label="predicct")
plt.plot(range(len(testPredictions)),labelsTest,'r',label="test")
plt.legend(loc="upper right")
plt.show()

##十折交叉验证
#nxval = 10
#for ixval in range(nxval):  #0-9
#    #定义测试和训练索引集
#    idxTest  = [a for a in range(nrow) if a%nxval == ixval%nxval]  #%nxval
#    idxTrain = [a for a in range(nrow) if a%nxval != ixval%nxval]
#    #定义测试和训练属性和标签集
#    xTrain = np.array([xList[r] for r in idxTrain])
#    xTest  = np.array([xList[r] for r in idxTest])
#    labelsTrain = np.array([labels[r] for r in idxTrain])
#    labelsTest  = np.array([labels[r] for r in idxTest])

#    #训练线性回归模型
#    LinearRegressionModel = linear_model.LinearRegression()
#    LinearRegressionModel.fit(xTrain,labelsTrain)

#    #对样本外数据进行预测
#    testPredictions = LinearRegressionModel.predict(xTest)

#    #画出预测值与实际值对比图
#    plt.figure()
#    plt.plot(range(len(testPredictions)),testPredictions,'b',label="predicct")
#    plt.plot(range(len(testPredictions)),labelsTest,'r',label="test")
#    plt.legend(loc="upper right")
#    plt.show()


#定义混淆矩阵函数
def confusionMatrix(predicted,actual,threshold):
    if(len(predicted)!=len(actual)):return -1
    tp=0.0
    fp=0.0
    tn=0.0
    fn=0.0
    for i in range(len(actual)):
        if(actual[i]>0.5):
            if(predicted[i]>threshold):
                tp += 1.0
            else:
                fn += 1.0
        else:
            if(predicted[i]<threshold):
                tn += 1.0
            else:
                fp += 1.0
    rtn=[tp,fn,fp,tn]
    return rtn

#------生成混淆矩阵，得出误分类率？为什么很低-----------
confusionMatTrain = confusionMatrix(testPredictions,yTest,0.5)      #[229.0, 3319.0, 12.0, 56296.0]
tp=confusionMatTrain[0]                                             #tp = 66.0 fn =3479.0 fp = 1.0 tn = 59952.0
fn=confusionMatTrain[1]
fp=confusionMatTrain[2]
tn=confusionMatTrain[3]
print("tp = "+ str(tp) +"\tfn =" + str(fn) +"\n" +"fp = "+ str(fp) +"\ttn = "+str(tn) +"\n")
print("误分类率为："+str((fn+fp)/(fn+fp+tp+tn)))     #误分类率为：0.055650227211975406

fpr,tpr,thresholds = roc_curve(yTest,testPredictions)
roc_auc = auc(fpr,tpr)
print('AUC for out_sample ROC curve:%f' %roc_auc)    #AUC for out_sample ROC curve:0.970672


#-------------------------------------------（2）十折交叉验证训练模型ElasticNet----------------------------
alpha = 1.0
#交叉验证折叠的数量
nxval = 10
for ixval in range(nxval):  #0-9
    #定义测试和训练索引集
    idxTest = [a for a in range(nrow) if a%nxval == ixval]  #%nxval
    idxTrain = [a for a in range(nrow) if a%nxval != ixval]
    #定义测试和训练属性和标签集
    xTrain = np.array([xList[r] for r in idxTrain])
    xTest = np.array([xList[r] for r in idxTest])
    labelsTrain = np.array([labels[r] for r in idxTrain])
    labelsTest = np.array([labels[r] for r in idxTest])
    #训练模型ElasticNet#用坐标下降法计算弹性网路径;弹性网络优化函数对于单输出和多输出而言是不同的。fit_intercept不需要计算插值项
    alphas, coefs, _ = linear_model.enet_path(xTrain,labelsTrain,l1_ratio=0.8, fit_intercept=False, return_models=False)
    #应用系数到测试数据上产生预测和积累
    if ixval == 0:                          #如果是第0次交叉验证：
        yOut = labelsTest                        #实际输出为：标签测试数据
        pred = np.dot(xTest, coefs)          #则预测结果为：属性测试数据与系数矩阵相点乘
    else:                                   #不是第0次的话，累积数据：
        #积累预测                            
        yTemp = np.array(yOut)                               
        yOut = np.concatenate((yTemp, labelsTest), axis=0)    #测试数据累积：沿着现有轴将labelTest加入数组yTemp中。
        #积累预测
        predTemp = np.array(pred)                            #预测数据累积：沿着现有轴将乘积加入数组predTemp中。
        pred = np.concatenate((predTemp, np.dot(xTest, coefs)), axis = 0)
alphas.shape    #(100,) #正则化路径中alpha的数量，alpha取100个不同的值
coefs.shape     #(26, 100)
pred.shape      #(179568, 100)

#--------------------计算误分类率------------------------------
misClassRate = []
_,nPred = pred.shape                    #179568*100
for iPred in range(1, nPred):           #1-99
    predList = list(pred[:, iPred])     #取预测数据集第iPred列的所有数据
    errCnt = 0.0
    for irow in range(nrow):            #0-207
        if (predList[irow] < 0.0) and (yOut[irow] >= 0.0):  #如果第i行i列预测值数据<0，且实际值>=0，则errCnt+1
            errCnt += 1.0
        elif (predList[irow] >= 0.0) and (yOut[irow] < 0.0):#如果第i行i列预测值数据>=0，且实际值<0，则errCnt+1
            errCnt += 1.0
    misClassRate.append(errCnt/nrow)    #误分类率为：errCnt/行数208
#找到画图和输出的最低点
minError = min(misClassRate)            #0.4683184086251448
idxMin = misClassRate.index(minError)   #0,对应alpha的取值为：0.056091188844527587
plotAlphas = list(alphas[1:len(alphas)]) #alpha从1-99

#画图：误分类率随alpha值的变化曲线
plt.figure()
plt.plot(plotAlphas, misClassRate, label='Misclassification Error Across Folds', linewidth=2)
plt.axvline(plotAlphas[idxMin], linestyle='--',label='CV Estimate of Best alpha')
plt.legend()
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Misclassification Error')
plt.axis('tight')
plt.show()

#------------------计算AUC----------------------
idxPos = [i for i in range(nrow) if yOut[i] > 0.0]  #标签值>0.0的行下标
yOutBin = [0] * nrow
for i in idxPos: yOutBin[i] = 1                     #标签值>0.0的改为1，<=0.0的改为0
auc_ = []
for iPred in range(1, nPred):                       #1-99
    predList = list(pred[:, iPred])                 #取预测数据集第iPred列的所有数据
    #根据预测分数计算接收者工作特征曲线（ROC AUC）下的面积。注意：此实现仅限于二进制分类任务或标签指示符格式的多标签分类任务。
    aucCalc = roc_auc_score(yOutBin, predList)      #真实值：yOutBin，预测值：predList
    auc_.append(aucCalc)    #99
#找出auc的最大值，以及对应索引值。以及最佳alpha值：
maxAUC = max(auc_)                                  #0.96653704620191805
idxMax = auc_.index(maxAUC)                         #98,对应alpha的取值为：6.4491236754171026e-05
alphas[idxMax]

#画图：auc随alpha值的变化曲线
plt.figure()
plt.plot(plotAlphas, auc_, label='AUC Across Folds', linewidth=2)
plt.axvline(plotAlphas[idxMax], linestyle='--',label='CV Estimate of Best alpha')
plt.legend()
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Area Under the ROC Curve')
plt.axis('tight')
plt.show()

#为什么差距如此之大？？？
print('Best Value of Misclassification Error = ', misClassRate[idxMin]) #误分类率的最佳值： 0.4683184086251448
print('Best alpha for Misclassification Error = ', plotAlphas[idxMin])  #最佳误分类率对应的alpha值： 0.0560911888445

print('Best Value for AUC = ', auc_[idxMax])                            #AUC的最佳值:  0.966537046202
print('Best alpha for AUC   =  ', plotAlphas[idxMax])                   #最佳AUC对应的alpha值: 6.01447432416e-05

