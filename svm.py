import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Training points
class1_points = np.array([[2, 3], [1, 4], [2, 7], [3, 3]])
class2_points = np.array([[5, 6], [6, 2], [7, 4], [8, 6]])

# Labels
class1_labels = np.ones(len(class1_points))  # +1 for class1
class2_labels = -np.ones(len(class2_points)) # -1 for class2

# Concatenate points and labels
X = np.concatenate((class1_points, class2_points))
y = np.concatenate((class1_labels, class2_labels))

# Create SVM model (with opmized Lagrangian function)
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# Get weights
weights = clf.coef_[0]
print('Weights:', weights)

# Get bias
bias = clf.intercept_[0]
print('Bias:', bias)


# Get support vectors with indices , use weights and bias to calculate support vectors
support_vector_indices = clf.support_
support_vectors = X[support_vector_indices]
print('Support Vectors:', support_vectors)

# Get number of support vectors for each class
print('Number of support vectors for each class:', clf.n_support_)
print('Indices of support vectors:', support_vector_indices)

# Get indices of support vectors
print('Indices of support vectors:', support_vector_indices)

# Get indices of support vectors for each class
print('Indices of support vectors for each class:', clf.support_)


# Plotting
plt.scatter(class1_points[:, 0], class1_points[:, 1], color='red', label='Class 1')
plt.scatter(class2_points[:, 0], class2_points[:, 1], color='blue', label='Class 2')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='green', label='Support Vectors')
plt.legend()

# Plot the decision boundary
a = -weights[0] / weights[1]
xx = np.linspace(0, 10)
yy = a * xx - bias / weights[1]
plt.plot(xx, yy, 'k-')

plt.show()

# Predict
