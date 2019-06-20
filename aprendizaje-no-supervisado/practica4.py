import numpy as np
import random

a = 3
i = 0
vectorA = random.sample(range(1, 20), 6)
input = []
for x in range(0, 20):
    rect = []
    for a in vectorA:
        i += 1
        rect.append(np.random.uniform(-a, a))
    input.append(np.array(rect))
input = np.array(input)

print(input)
weights = []
for x in range(0, 4):
    weights.append(np.random.uniform(-0.5, 0.5, 6))
weights = np.array(weights)
print(weights)

learning_rate = 0.05
tolerance = 0.1
average_weight_change = 100
epsilon = 0.01
iterations = 0

while average_weight_change > epsilon and iterations < 5000:
    iterations += 1
    random.shuffle(input)
    diff_weights = 0
    for data in input:
        data = np.reshape(data, (data.size, 1))
        signal = np.dot(data.transpose(), weights.transpose())
        # sum sk * wkj
        new_weights = np.zeros((4, 6))
        for j, x in enumerate(data):
            for i in range(0, 4):
                sum = 0
                for k in range(0,i):
                    sum += signal[0][k] * weights[k][j]
                new_weights[i][j] = weights[i][j] + signal[0][i] * learning_rate / iterations * (x - sum)
        #new_weights = weights + learning_rate / iterations * signal * (data-weights_by_signal)
        diff_weights += np.median(weights - new_weights)
        print("{1} - DIFF: {0}".format(diff_weights, iterations))
        weights = new_weights
    average_weight_change = diff_weights / len(input)
    print("{1} - Average weight change: {0}".format(average_weight_change, iterations))
