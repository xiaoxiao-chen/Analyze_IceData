#采用岭回归、套索回归、ElasticNet回归模型对数据进行预测，并且根据属性重要性进行排序。

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_curve,auc

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

#----将列表转化为数组，用于训练模型---
X = np.array(xList)
Y = np.array(labels)

#----------------------------------(1)岭回归（十折交叉验证）--------------
alphaList = [0.1**i for i in [-4,-3,-2,-1,0,1,2,3,4,5]]
RidgeModelCV = linear_model.RidgeCV(cv=10,alphas=alphaList)     # fit_intercept=True,？？？设置为False反而出错？？
RidgeModelCV.fit(X,Y)
Ridge_alphaBest = RidgeModelCV.alpha_     #999.9999999999999
Ridge_coefStar = RidgeModelCV.coef_
Ridge_absCoef = [abs(a) for a in Ridge_coefStar]
Ridge_sortCoef = sorted(Ridge_absCoef,reverse=True)
Ridge_idxCoef = [Ridge_absCoef.index(a) for a in Ridge_sortCoef if not(a==0.0)]
Ridge_nameList = [names[Ridge_idxCoef[i]] for i in range(len(Ridge_idxCoef))]


#-----------------------------------(2)套索回归----------------------------
#按最佳alpha时的系数大小排序
LassoModel = linear_model.LassoCV(cv=10)
LassoModel.fit(X,Y)
Lasso_alphaBest = LassoModel.alpha_   #0.0005533404800913608
Lasso_coefStar = LassoModel.coef_
Lasso_absCoef = [abs(a) for a in Lasso_coefStar]
Lasso_sortCoef = sorted(Lasso_absCoef,reverse=True)
Lasso_idxCoef = [Lasso_absCoef.index(a) for a in Lasso_sortCoef if not(a==0.0)]
Lasso_nameList = [names[Lasso_idxCoef[i]] for i in range(len(Lasso_idxCoef))]


#按进入模型的时间排序
alphas_lasso,coefs_lasso,_=linear_model.lasso_path(X,Y,retutn_models=False)
nattr_lasso,nalpha_lasso = coefs_lasso.shape     #26*100
nzList_lasso = []
for iAlpha in range(1,nalpha_lasso):
    coefList = list(coefs_lasso[:,iAlpha])
    nzCoef = [index for index in range(nattr) if coefList[index]!=0.0]
    for q in nzCoef:
        if not(q in nzList_lasso):
            nzList_lasso.append(q)
namesList_lasso = [names[nzList_lasso[i] ]for i in range(len(nzList_lasso))]

#LassoLarsCV 按最佳alpha时的系数大小排序
LassoLarsModel = linear_model.LassoLarsCV(cv=10)
LassoLarsModel.fit(X,Y)
LassoLars_alphaBest = LassoLarsModel.alpha_   #1.7880613340934534e-06
LassoLars_coefStar = LassoLarsModel.coef_
LassoLars_absCoef = [abs(a) for a in LassoLars_coefStar]
LassoLars_sortCoef = sorted(LassoLars_absCoef,reverse=True)
LassoLars_idxCoef = [LassoLars_absCoef.index(a) for a in LassoLars_sortCoef if not(a==0.0)]
LassoLars_nameList = [names[LassoLars_idxCoef[i]] for i in range(len(LassoLars_idxCoef))]


#------------------------------------(3)ElasticNet--------------
#按最佳alpha时的系数大小排序
ElasticNetModel = linear_model.ElasticNetCV(cv=10,l1_ratio=0.8)
ElasticNetModel.fit(X,Y)
ElasticNet_alphaBest = ElasticNetModel.alpha_   #0.00069167560011420073
ElasticNet_coefStar = ElasticNetModel.coef_
ElasticNet_absCoef = [abs(a) for a in ElasticNet_coefStar]
ElasticNet_sortCoef = sorted(ElasticNet_absCoef,reverse=True)
ElasticNet_idxCoef = [ElasticNet_absCoef.index(a) for a in ElasticNet_sortCoef if not(a==0.0)]
ElasticNet_nameList = [names[ElasticNet_idxCoef[i]] for i in range(len(ElasticNet_idxCoef))]


#按进入模型的时间进行排序
alphas_ElasticNet,coefs_ElasticNet,_=linear_model.enet_path(X,Y,l1_ratio=0.8,fit_intercept=False,retutn_models=False)
nattr_ElasticNet,nalpha_ElasticNet = coefs_ElasticNet.shape     #26*100
nzList_ElasticNet = []
for iAlpha in range(1,nalpha_ElasticNet):
    coefList = list(coefs_ElasticNet[:,iAlpha])
    nzCoef = [index for index in range(nattr_ElasticNet) if coefList[index]!=0.0]
    for q in nzCoef:
        if not(q in nzList_ElasticNet):
            nzList_ElasticNet.append(q)
namesList_ElasticNet = [names[nzList_ElasticNet[i] ]for i in range(len(nzList_ElasticNet))]



#---------------------------------------（4）前向逐步回归---------------------------
indices = range(xList)
xListTest  = [xList[i] for i in indices if i%3 == 0 ]
xListTrain = [xList[i] for i in indices if i%3 != 0 ]
labelsTest  = [labels[i] for i in indices if i%3 == 0 ]
labelsTrain = [labels[i] for i in indices if i%3 != 0 ]

xTrain = np.array(xListTrain)
xTest  = np.array(xListTest)
yTrain = np.array(labelsTrain)
yTest  = np.array(labelsTest)

def xattrSelect(x, idxSet):
    #将X矩阵作为list of list并返回包含idxSet中的列的子集
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return(xOut)

attributeList = []              #最佳属性索引列表
index = range(len(xList[1]))    #（0，26）
indexSet = set(index)           #{0，1，......25}
indexSeq = []
#逐步前向回归求auc最大时的属性排序
AUCList = []
for i in index:                         #0-25
    attSet = set(attributeList)         #最佳属性索引集,第一次循环时为空
    attTrySet = indexSet-attSet         #前面循环得到的索引列表不在该列表集中
    attTry = [ii for ii in attTrySet]   #属性列表集转换为属性列表，未检查的属性列表
    aucList = []                        #auc列表
    attTemp = []                        #属性子集
        
    #设置每个属性，看哪一个auc值最大
    for iTry in attTry:
        attTemp = []+ attributeList     #属性子集每次循环存放上一个大循环中得出的误差最小的属性列表
        attTemp.append(iTry)            #每次小循环都加入一个属性
        #用attTemp作为list of list，形成训练和测试子矩阵,从xListTrain中提取attTemp列作为xTrainTemp
        xTrainTemp = xattrSelect(xListTrain,attTemp)
        xTestTemp  = xattrSelect(xListTest,attTemp)
        xTrain = np.array(xTrainTemp)
        xTest  = np.array(xTestTemp)
    
        #使用sci-kit学习训练线性回归模型
        LinearRegressionModel = linear_model.LinearRegression()
        LinearRegressionModel.fit(xTrain,yTrain)
        
        #使用训练好的模型生成预测并计算auc
        fpr,tpr,thresholds = roc_curve(yTest,LinearRegressionModel.predict(xTest))
        roc_auc = auc(fpr,tpr)
        aucList.append(roc_auc)
        attTemp = []    #将属性子集列表置为空，循环下一次遍历
    iBest = np.argmax(aucList)                 #argmax()函数返回沿轴的最大值的索引。
    attributeList.append(attTry[iBest])
    AUCList.append(aucList[iBest])
print("the value of auc：" )
print(AUCList)
print("\n" + "最佳属性索引")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\n" + "最佳属性名字")
print(namesList)                                  

#----------------------------------------LARS算法--------------------------
#--初始化系数β的向量--初始化存放每一步得到的β
beta = [0.0] * ncol
betaMat = []
betaMat.append(list(beta))
#步数和步长
nSteps = 350
stepSize = 0.004
nzList = []
for i in range(nSteps):             #0-349
    #计算残差
    residuals = [0.0] * nrow
    for j in range(nrow):           #0-179568
        labelsHat = sum([xList[j][k] * beta[k] for k in range(ncol)])    #0-25
        residuals[j] = labels[j] - labelsHat
    #计算已经归一化的属性列与残差之间的相关性
    corr = [0.0] * ncol
    for j in range(ncol):
        corr[j] = sum([xList[k][j] * residuals[k] for k in range(nrow)]) / nrow
    iStar = 0
    corrStar = corr[0]
    #找出|corr|最大的属性列
    for j in range(1, (ncol)):
        if abs(corrStar) < abs(corr[j]):
            iStar = j; corrStar = corr[j]
    #判断关联系数为正，则增加，为负则减小
    beta[iStar] += stepSize * corrStar / abs(corrStar)
    betaMat.append(list(beta))
    #将关联性最强的属性列下标加入列表nzList
    nzBeta = [index for index in range(ncol) if beta[index] != 0.0]
    for q in nzBeta:
        if (q in nzList) == False:
            nzList.append(q)
#找出按关联性排列的属性列的名称顺序
nameList = [names[nzList[i]] for i in range(len(nzList))]
print(nameList)
print(nzList)
