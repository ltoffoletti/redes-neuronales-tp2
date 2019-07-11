import numpy as np
import pandas as pd
from ClaseOjaSangerEmilio import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#TODO cambiar a relative PATH
def load_data(filename= "/home/luis/PycharmProjects/redesNeuronales/tp2/data/tp2_training_dataset.csv"):
    data = pd.read_csv(filename, header=None)
    # return data.drop(axis=1, labels=0)
    return data




#carga datos de entrada sin la categoria (primera columna)
inputX = np.array(load_data())[:,0:1]
input = np.array(load_data())[:,1:]
print(input)

input2 = np.array([[-1,0], [1, 0], [0, 0], [0.5, 0], [-0.5, 0], [0, -0.5], [0, 0.5]])


pca = PCA()
pca.fit(input.T)

colors = ['black', 'blue', 'purple', 'red', 'yellow', 'orange', 'green', 'pink', "gray"]
for i, x in enumerate(inputX):
    plt.scatter(pca.components_[i][0], pca.components_[i][1], c=colors[x[0]-1])
plt.show()
# input_number = 850
# output_number = 3

ros = redOjaSanger()
n = 2
L_in = input2
learning_rate = 0.015
w_tolerance = 0.001
w_init=50

ros.OjaSanger(n,L_in,n_train=learning_rate,w_tolerance=w_tolerance,w_init=w_init,OjaSanger=1)

