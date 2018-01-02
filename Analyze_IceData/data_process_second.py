#-------------------------------------------------------------------------------------
#读取处理后的正确数据集，进行一系列的初步数据分析

import pandas as pd
from pandas import DataFrame
import gc 
from math import sqrt 
from matplotlib import pyplot as plt
import numpy as np


#-------------读取正确数据集-------------
Ice_data = pd.read_csv("data/train/21/21_newdata.csv")

#---------------------------------------(1)确定数据集的物理特性--------------------------

#-------确定数据集的规模：行数、列数----
nrow = len( Ice_data.index )
ncol = len( Ice_data.columns)

print("\nNumber of Rows of Data: %d" % nrow + "\n")     #179568
print("\nNumber of Cols of Data: %d" % ncol + "\n")     #29

#-------确定每个属性值的特性-----------
print(Ice_data.head())
print(Ice_data.tail())

#通过输出数据集的首尾各五行数据可以看出，第0列数据为time，第1-26列为属性值，为连续性数值变量
#第27列为分组标识，第28列为标签值（二值类别变量）。

#--------------------------------------（2）将数据分成属性列表和标签列表--------------------------
xList = []      #属性列表
labels = []     #标签列表
names = []      #列名，各属性及标签名称

names1 = Ice_data.columns.values.tolist()
names = names1[1:-2]                            #排除第0列time,最后两列
labels = Ice_data.iloc[:,-1].values.tolist()    
xList = Ice_data.iloc[:,1:-2].values.tolist()   #排除第0列time，第27列group,方便分析

#--------------------------------------（3）确定数据集的统计特性-----------------------------
#------输出数值型变量的统计信息(1-26列）---------
summary = Ice_data.iloc[:,1:ncol-2].describe()      #8行26列

#存入文件21_summary.csv
summary.to_csv("data/train/21/21_summary.csv")
print(summary)

#------确定类别型属性具体类别的数量分布-----------
count = [0]*2
for i in range(len(labels)):
    if(labels[i] == 0):
        count[0] += 1
    else:
        count[1] += 1
gc.collect()

#计算分类标签的每个值#[168930, 10638]不结冰-结冰  
print("\nCounts for Each Value of Categorical Label: "+ str(count)+"\n")   

#正例百分比10638\179568=0.059242181234963914
print("\nThe positive percentage is: " + str(count[1]/nrow) +"\n")     

#-------------------------------------（4）将所有数据进行归一化-----------------------------------
#---------------归一化属性列----------
xList_Normalized=Ice_data.iloc[:,1:-2]           #(1,27)1-26
nrows = len(xList_Normalized)                    #行数：179568
ncols = len(xList_Normalized.columns)            #列数：26

for i in range(ncols):                          
    mean = summary.iloc[1,i]
    sd = summary.iloc[2,i]
    xList_Normalized.iloc[:,i:(i+1)] = ( xList_Normalized.iloc[:,i:(i+1)]-mean ) / sd
xNormalized = xList_Normalized.values.tolist()

print("\nNumber of Rows of Normalized Data: %d" % nrows + "\n")         
print("\nNumber of Columnss of Normalized Data: %d" % ncols +"\n")     

#--------------不归一化标签列--------------
#meanLabel = sum(labels)/nrow
#sdLabel = sqrt(sum([(labels[i]-meanLabel)*(labels[i]-meanLabel) for i in range(nrows)])/nrows)
#labelNormalized = [(labels[i]-meanLabel)/sdLabel for i in range(nrows)]


#将归一化之后的数据存入文件,(179568, 27)
Labels  = DataFrame(labels)
Data_Normalized = pd.concat([xList_Normalized,Labels],axis=1)
Data_Normalized.columns = names1[1:-1]
Data_Normalized.columns.values[-1]=names1[-1]
Data_Normalized.to_csv('data/train/21/21_Normalized_Data.csv',index=False)

#-------------------------------------（5）数据集属性的可视化展示-----------------------------------

#------------画出数值型数据的箱线图，显示异常点-------
array = np.array(xNormalized)
plt.boxplot(array,sym ="o", whis = 1.5)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
plt.show()

#------------画出平行坐标图，显示出属性之间的关系-----
for i in range(nrows):          #行数太多，无法全部画出，只能分开画图
    if xList[i][-1] == 1:
        pcolor = "red"
    else:
        pcolor = "blue"
    dataRow = xList_Normalized
    dataRow.plot(color=pcolor,alpha=0.5)
plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

#------------画出热图，展示属性之间的相关性-----------
corMat = DataFrame(xList_Normalized.corr())
plt.pcolor(corMat)
plt.show()

#将属性之间的相关系数写入文件corMat.csv    
corMat.to_csv("data/train/21/21_corMat.csv")
