from minisom import MiniSom
import numpy as np
dim=3
data_file = np.load('input_som/3pcs/CP-w_tolerance0.0001l_rate0.02OjaSanger1w_init50pcs3.npz.npy')
docu_file = data_file[:, :dim]
categoria = data_file[:, dim]
som = MiniSom(10, 10, 3, sigma=0.5, learning_rate=0.5)
som.train_random(docu_file, 100)
