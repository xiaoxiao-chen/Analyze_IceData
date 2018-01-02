#------------------------------------------------------------------------------------- 
#剔除无效数据，根据结冰不结冰开始结束时间判断当前状态，结冰：1，不结冰：0 
 
#对原始数据进行整合，输入几个原始数据文件，输出处理后的数据文件 
 
import pandas as pd 
from pandas import DataFrame  
 
##定义一个函数，判断时刻t时，是否结冰。对列表t_Ice的结冰范围的值改为1，其余部分值为0.1 
def t_IceOrNot_Failure(time,Ice_startTime,Ice_endTime,t_Ice): 
    for i in range(len(time)-1,-1,-1):              #393885---0 
        for j in range(len(Ice_startTime)-1,-1,-1): #26-------0 
            if(Ice_startTime[j]<=time[i]<=Ice_endTime[j]): 
                t_Ice[i]=1 
                break 
    return t_Ice 
 
def t_IceOrNot_Normal(time,NotIce_startTime,NotIce_endTime,t_Ice): 
    for i in range(len(time)-1,-1,-1):              #393885---0 
        for j in range(len(NotIce_startTime)-1,-1,-1): #28-------0 
            if(NotIce_startTime[j]<=time[i]<=NotIce_endTime[j]): 
                t_Ice[i]=0 
                break 
    return t_Ice 
 
#定义一个函数，输入原始数据、结冰时间数据、不结冰时间数据，输出整合好的数据 
def source_to_aimsdata(data,failureInfo,normalInfo,newdata): 
    data = pd.read_csv(data) 
    failureInfo = pd.read_csv(failureInfo) 
    normalInfo = pd.read_csv(normalInfo) 
 
    time = data.loc[:,"time"] 
    Ice_startTime = failureInfo.loc[:,'startTime'] 
    Ice_endTime = failureInfo.loc[:,'endTime'] 
    NotIce_startTime = normalInfo.loc[:,'startTime'] 
    NotIce_endTime = normalInfo.loc[:,'endTime'] 

    #定义一个list，值为1或0，代表t时刻-结冰或者不结冰，默认为0.1 
    t_Ice=[0.1 for i in range(1) for row in range(len(time))] 
 
    #对列表t_Ice的结冰范围的值改为1,不结冰范围值为0，其余为0.1，把该无效数据剔除 
    t_Ice_1 = t_IceOrNot_Failure(time,Ice_startTime,Ice_endTime,t_Ice) 
    t_Ice_2 = t_IceOrNot_Normal(time,NotIce_startTime,NotIce_endTime,t_Ice)  
 
    #------将原有数据与是否结冰I数据进行合并，横向------ 
    data_Ice =DataFrame(t_Ice_2) 
    data_new = pd.concat([data, data_Ice], axis=1) 
    data_new.columns.values[28]='ice_or_not'                 #添加表头 
 
    #----剔除最后一列值为0.1的无效数据----- 
    indexs = [] 
    for i in range(len(data_new.index)): 
        if(data_new.iloc[i,-1] == 0.1): 
            indexs.append(i) 
        else: 
            continue 
    #剔除的无效数据的行数为：10926 
    print("Invalid data has %d lines" % len(indexs)) 
 
    #剔除最后一列值为0.1的行索引代表的行 
    data_new.drop(indexs,inplace=True) 
 
    #写入文件aims    #不要索引和表头 
    data_new.to_csv(newdata,index=False) 
 
    #读取处理后的正确数据集Ice_data 
    aimsdata = pd.read_csv(newdata) 
    return aimsdata 
 
#-----------------------------#读取处理后的正确数据集Ice_data---------------------------------------------- 
Ice_data = source_to_aimsdata("data/train/21/21_data.csv","data/train/21/21_failureInfo.csv",\
    "data/train/21/21_normalInfo.csv","data/train/21/21_newdata.csv") 
