X = data['data'][:1000, 0]
y = data['target'][:1000]
# X
bias = 0
n_samples = 1000
# plt.scatter(X, y)
n_features = 1
weights = np.zeros(n_features)
lr = 0.01
n_samples
# Training
epochs = 1000
for _ in range(epochs):
    for i in range(len(X)):
        y_pred = np.dot(X[i], weights) + bias
        
        dw = (1 / n_samples) * np.dot(X[i].T, (y_pred - y[i]))  # X_Transpose
        db = (1 / n_samples) * np.sum(y_pred - y[i])
        
        weights = weights - lr * dw
        bias = bias - lr * db 
# Testing
ans = []
for i in range(len(X)):
    ans.append(np.dot(X[i], weights)+ bias)
# ans
fig = plt.figure(figsize=(10, 5))
plt.scatter(X, y)
plt.plot(X, ans, color = 'green')
# y.shape
