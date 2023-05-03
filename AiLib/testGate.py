import numpy as np
import pandas as pd
import build.AIBoost as aib
from build.AIBoost import DataSet
from build.AIBoost import Model
from build.AIBoost import Layer

df_train = pd.read_csv("/home/musasina/Desktop/AI_Boost/and.csv",delimiter = ",").sample(frac=1)
train_data = df_train.iloc[:,:2].values.astype(float).tolist()
train_result = df_train.iloc[:,2].values.astype(float).tolist()
train_result = np.array(train_result,dtype=float).reshape((3,1)).tolist()

train_data_vec = aib.FloatVectorVector()
train_data_vec.reserve(3)
train_result_vec = aib.FloatVectorVector()
train_result_vec.reserve(3)

for row in train_data:
    tmp = aib.FloatVector()
    tmp.reserve(len(row))
    for col in row:
        tmp.push_back(float(col))
    train_data_vec.push_back(tmp)
for row in train_result:
    tmp = aib.FloatVector()
    tmp.reserve(len(row))
    for col in row:
        tmp.push_back(float(col))
    train_result_vec.push_back(tmp)

dataset = DataSet(train_data_vec,train_result_vec,3,2,3,1)

tanhF = aib.Tanh
sigmoid = aib.Sigmoid
l1reg = aib.L1
CELoos = aib.CELoss
BCELoss = aib.BCELoss

layer1 = Layer(2,3,tanhF)
layer3 = Layer(3,1,sigmoid)

layer_list = aib.LayerPtrVector()
for i in [layer1,layer3]:
    layer_list.push_back(i)
model = Model(layer_list,dataset,False,42,0.0,l1reg,0.0,BCELoss,0.001,True,1,2)

model.trainModel()