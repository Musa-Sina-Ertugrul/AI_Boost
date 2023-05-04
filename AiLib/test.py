
import numpy as np
import pandas as pd
import build.AIBoost as aib
from build.AIBoost import DataSet
from build.AIBoost import Model
from build.AIBoost import Layer
import ctypes
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("/home/musasina/Desktop/AI_Boost/train.csv",delimiter = ",")
train_data = df_train.iloc[:42000,1:].values.astype(float).tolist()
train_result = df_train.iloc[:42000,0].values.astype(float).tolist()
train_result = np.array(train_result,dtype=float).reshape((42000,1)).tolist()
train_result_list = []
for i in range(42000):
    num = train_result[i][0]
    train_result_list.append([1.0 if j == num else 0.0 for j in range(10)])
train_data_vec = aib.FloatVectorVector()
train_data_vec.reserve(33600)
train_result_vec = aib.FloatVectorVector()
train_result_vec.reserve(33600)
x_train,x_test,y_train,y_test = train_test_split(train_data,train_result_list,test_size=0.2,random_state=0,shuffle=True)

for row in x_train:
    tmp = aib.FloatVector()
    tmp.reserve(len(row))
    for col in row:
        tmp.push_back(float(col))
    train_data_vec.push_back(tmp)
for row in y_train:
    tmp = aib.FloatVector()
    tmp.reserve(len(row))
    for col in row:
        tmp.push_back(float(col))
    train_result_vec.push_back(tmp)
dataset = DataSet(train_data_vec,train_result_vec,33600,784,33600,10)
Relu = aib.Relu
softmax= aib.SoftMax
sigmod = aib.Sigmoid
l1reg = aib._None
tanH = aib.Tanh

CELoos = aib.CELoss
l1Loss = aib.L1Loss
l2Loss = aib.L2Loss
layer1 = Layer(784,400,tanH)
layer2 = Layer(400,400,tanH)
layer3 = Layer(400,10,softmax)
layer_list = aib.LayerPtrVector()
for i in [layer1,layer2,layer3]:
    layer_list.push_back(i)
model = Model(layer_list,dataset,False,1000,0.0,l1reg,0.0,CELoos,0.001,True,100,3)

model.trainModel()

