
import numpy as np
import pandas as pd
import build.AIBoost as aib
from build.AIBoost import DataSet
from build.AIBoost import Model
from build.AIBoost import Layer
import ctypes
import math

df_train = pd.read_csv("/home/musasina/Desktop/AI_Boost/train.csv",delimiter = ",")
train_data = df_train.iloc[:10000,1:].values.astype(float).tolist()
train_result = df_train.iloc[:10000,0].values.astype(float).tolist()
train_result = np.array(train_result,dtype=float).reshape((10000,1)).tolist()
train_result_list = []
for i in range(10000):
    num = train_result[i][0]
    train_result_list.append([1.0 if j == num else 0.0 for j in range(10)])
train_data_vec = aib.FloatVectorVector()
train_data_vec.reserve(10000)
train_result_vec = aib.FloatVectorVector()
train_result_vec.reserve(10000)
for row in train_data:
    tmp = aib.FloatVector()
    tmp.reserve(len(row))
    for col in row:
        tmp.push_back(float(col))
    train_data_vec.push_back(tmp)
for row in train_result_list:
    tmp = aib.FloatVector()
    tmp.reserve(len(row))
    for col in row:
        tmp.push_back(float(col))
    train_result_vec.push_back(tmp)
dataset = DataSet(train_data_vec,train_result_vec,10000,784,10000,10)
tanhF = aib.Tanh
softmax = aib.SoftMax
l1reg = aib.L1
CELoos = aib.CELoss

layer1 = Layer(784,40,tanhF)
layer2 = Layer(40,40,tanhF)
layer3 = Layer(40,10,softmax)
layer_list = aib.LayerPtrVector()
for i in [layer1,layer2,layer3]:
    layer_list.push_back(i)
model = Model(layer_list,dataset,True,10000,0.0,l1reg,0.001,CELoos,0.001,True,1000,3)

model.trainModel()
