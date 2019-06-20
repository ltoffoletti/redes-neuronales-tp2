import numpy as np
import random
import matplotlib.pyplot as plt
#init

#uniforme + normal

input_y = np.random.uniform(-0.01, 0.01, 200)
input_x = np.random.normal(0, 0.5, 200)

input = list(zip(input_x, input_y))
plt.scatter(input_x, input_y)
learning_rate = 0.05
weights = np.array([0.1, 0.5])

tolerance = 0.1
average_weight_change = 100
epsilon = 0.01
iterations = 0
while average_weight_change > epsilon and iterations < 5000:
    iterations += 1
    random.shuffle(input)
    plt.scatter(weights[0], weights[1], marker='^')
    diff_weights = 0
    for data in input:
        signal = np.dot(data, weights)
        weights_by_signal = np.array([(signal * w) for w in weights])
        new_weights = weights + learning_rate * signal * (data-weights_by_signal)
        diff_weights += abs(sum(weights - new_weights))
        weights = new_weights
    average_weight_change = diff_weights / len(weights)
    print("{1} - Average weight change: {0}".format(average_weight_change, iterations))
plt.show()