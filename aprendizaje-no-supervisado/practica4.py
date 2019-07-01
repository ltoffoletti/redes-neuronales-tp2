import numpy as np
import random

input_number = 2
output_number = 4
vectorA = random.sample(range(1, 20), input_number)
input = []
i = 0
for x in range(0, 20):
    rect = []
    for a in vectorA:
        i += 1
        rect.append(np.random.uniform(-a, a))
    input.append(np.array(rect))
input = np.array(input)

print(input)
weights = []
for x in range(0, output_number):
    weights.append(np.random.uniform(-0.5, 1.5, input_number))
weights = np.array(weights)
print(weights)

learning_rate = 0.0001
tolerance = 0.1
average_weight_change = 100
epsilon = 0.0000000001
iterations = 0
n = learning_rate
while average_weight_change > epsilon and iterations < 5000:
    iterations += 1
    random.shuffle(input)
    diff_weights = 0
    n = n - n * learning_rate * epsilon * iterations
    for data in input:
        data = np.reshape(data, (data.size, 1))
        signal = np.dot(data.transpose(), weights.transpose())
        # sum sk * wkj
        new_weights = np.zeros((output_number, input_number))
        for j, x in enumerate(data):
            for i in range(0, output_number):
                sum = 0
                for k in range(0,i):
                    sum += signal[0][k] * weights[k][j]
                new_weights[i][j] =
                # new_weights[i][j] = weights[i][j] + signal[0][i] * (n) * (x - sum)
        #new_weights = weights + learning_rate / iterations * signal * (data-weights_by_signal)
        diff_weights += np.abs(np.median(weights - new_weights))
        print("{1} - DIFF: {0}".format(diff_weights, iterations))
        weights = new_weights
    average_weight_change = diff_weights / len(input)
    print("{1} - Average weight change: {0}".format(average_weight_change, iterations))
