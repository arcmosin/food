import matplotlib.pyplot as plt
import numpy as np
import csv

file='Adam'
path_trainLoss=file+'/trainLossData.csv'
path_validLoss=file+'/validLossData.csv'

data_train=np.loadtxt(path_trainLoss)
data_valid=np.loadtxt(path_validLoss)
print(len(data_train))
x=range(len(data_train))

def show_data():
    plt.figure(figsize=(10,5))
    plt.title('loss in train and valid')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    #plt.plot(x,data_train) #无标签简单绘图
    #plt.plot(x,data_valid)
    plt.plot(x,data_train,'-',label='train')
    plt.plot(x, data_valid, '-', label='valid')
    plt.legend()
    plt.show()

show_data()