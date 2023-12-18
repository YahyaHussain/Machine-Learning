X = np.array([ [1, 0], [1, 1] ])
y = np.array([1, 0])

weights = np.array([0.5, 0.8])
bias = 0.7
lr = 0.1

def activation(z):
    return 1 if z > 0 else 0

epoch = 100
for i in range(epoch):
    for j in range(len(X)):
        temp = np.dot(X[j], weights) + bias
        prediction = activation(temp)
        weights = weights + lr * (y[j] - prediction) * X[j]
        bias = bias + lr * (y[j] - prediction)

for i in range(len(X)):
    print('Actual: ', y[i])
    print('Predicted: ', activation(np.dot(X[i], weights) + bias))
