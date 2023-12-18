import numpy as np 
X = np.array([ [0.2,  0.4], [0.4, 0.6], [0.6, 0.8] ])
y_true = np.array([0, 1, 1]) 
y_true[0] = 1.5 
X_b = np.c_[np.ones((X.shape[0], 1)), X]
initial_weights = np.random.randn(X_b.shape[1], 1)
learning_rate = 0.01
n_iterations = 1000
weights = initial_weights.copy()
for iteration in range(n_iterations):
errors = y_true - X_b.dot(weights).flatten()
gradients = -X_b.T.dot(errors.reshape(-1, 1))
weights = weights - learning_rate * gradients 

print("Initial Weights: ")
print(initial_weights)
print("\nerror")
print(errors)
print("\nUpdated Weights:")
print(weights)
