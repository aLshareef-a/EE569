from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants hyperparameters
CLASS1_SIZE = 100  # Number of samples in Class 1
CLASS2_SIZE = 100  # Number of samples in Class 2
N_FEATURES = 2     # Number of input features
N_OUTPUT = 1       # Number of output units (binary classification)
LEARNING_RATE = 0.001  # Learning rate for SGD
EPOCHS = 100       # Number of epochs for training
TEST_SIZE = 0.25   # Proportion of the dataset to be used as test data
BATCH_SIZE = 10    # Batch size for training

# Define the means and covariances of the two components to create an XOR problem
MEAN1 = np.array([0, 0])  # Mean of Gaussian distribution for Class 1
COV1 = np.array([[0.1, 0], [0, 0.1]])  # Covariance matrix for Class 1
MEAN2 = np.array([1, 1])  # Mean of Gaussian distribution for Class 2
COV2 = np.array([[0.1, 0], [0, 0.1]])  # Covariance matrix for Class 2
MEAN3 = np.array([0, 1])  # Mean of Gaussian distribution for Class 1
COV3 = np.array([[0.1, 0], [0, 0.1]])  # Covariance matrix for Class 1
MEAN4 = np.array([1, 0])  # Mean of Gaussian distribution for Class 2
COV4 = np.array([[0.1, 0], [0, 0.1]])  # Covariance matrix for Class 2

# Generate random points from the four components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE // 2)  # Samples for Class 1
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE // 2)  # Samples for Class 2
X3 = multivariate_normal.rvs(MEAN3, COV3, CLASS1_SIZE // 2)  # Samples for Class 1
X4 = multivariate_normal.rvs(MEAN4, COV4, CLASS2_SIZE // 2)  # Samples for Class 2

# Combine the points and generate labels
X = np.vstack((X1, X2, X3, X4))  # Combine samples into a single array
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))  # Labels: 0 for Class 1, 1 for Class 2

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data (XOR Problem)')
plt.show()

# Split data into training and test sets
indices = np.arange(X.shape[0])  # Create an array of indices
np.random.shuffle(indices)  # Shuffle indices

test_set_size = int(len(X) * TEST_SIZE)  # Calculate size of the test set
test_indices = indices[:test_set_size]  # Indices for the test set
train_indices = indices[test_set_size:]  # Indices for the training set

X_train, X_test = X[train_indices], X[test_indices]  # Split features into training and test sets
y_train, y_test = y[train_indices], y[test_indices]  # Split labels into training and test sets

# Model parameters
n_features = X_train.shape[1]  # Number of input features
n_output = 1  # Binary classification

# Initialize weights and biases for MLP
W0 = np.zeros((1, n_output))  # Output layer bias
W1 = np.random.randn(n_features, 20) * 0.1  # First hidden layer weights
W2 = np.random.randn(20, 20) * 0.1  # Second hidden layer weights
W3 = np.random.randn(20, n_output) * 0.1  # Output layer weights
B1 = np.zeros((1, 20))  # First hidden layer bias
B2 = np.zeros((1, 20))  # Second hidden layer bias

# Create nodes for the computational graph
x_node = Input()  # Input node for features
y_node = Input()  # Input node for labels

w1_node = Parameter(W1)  # First hidden layer weights
w2_node = Parameter(W2)  # Second hidden layer weights
w3_node = Parameter(W3)  # Output layer weights
b1_node = Parameter(B1)  # First hidden layer bias
b2_node = Parameter(B2)  # Second hidden layer bias
w0_node = Parameter(W0)  # Output layer bias

# Build the computational graph for MLP
u1_node = Linear(w1_node, x_node, b1_node)  # First hidden layer linear transformation
a1_node = Sigmoid(u1_node)  # First hidden layer activation
u2_node = Linear(w2_node, a1_node, b2_node)  # Second hidden layer linear transformation
a2_node = Sigmoid(u2_node)  # Second hidden layer activation
u3_node = Linear(w3_node, a2_node, w0_node)  # Output layer linear transformation
sigmoid = Sigmoid(u3_node)  # Sigmoid activation for output layer
loss = BCE(y_node, sigmoid)  # Binary Cross-Entropy loss node

# Create graph and trainable parameter list
graph = [x_node, w1_node, b1_node, u1_node, a1_node,
         w2_node, b2_node, u2_node, a2_node, w3_node, w0_node, u3_node, sigmoid, loss]  # Computational graph
trainable = [w1_node, b1_node, w2_node, b2_node, w3_node, w0_node]  # Parameters to be updated during training

# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()  # Compute forward pass for each node

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()  # Compute backward pass for each node

# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * np.mean(t.gradients[t], axis=0).reshape(t.value.shape)  # Update parameters

epoch_array = []
loss_array = []

# Training loop
for epoch in range(EPOCHS):
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        x_node.value = X_train[i:i + BATCH_SIZE]  # Assign batch of features to input node
        y_node.value = y_train.reshape(-1, 1)[i:i + BATCH_SIZE]  # Assign batch of labels to input node
        forward_pass(graph)  # Perform forward pass
        backward_pass(graph)  # Perform backward pass
        sgd_update(trainable, LEARNING_RATE)  # Update parameters
    epoch_array.append(epoch + 1)
    loss_array.append(loss.value)

    print(f"Epoch {epoch + 1}, Loss: {loss.value}")  # Log loss for each epoch

# Evaluate the model
correct_predictions = 0
x_node.value = X_test  # Assign test features to input node
y_node.value = y_test  # Assign test labels to input node
forward_pass(graph)  # Perform forward pass

# Count correct predictions
for i in range(X_test.shape[0]):
    if (sigmoid.value[i] >= 0.5) == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]  # Calculate accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize decision boundary
x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))

x_node.value = np.array(list(zip(xx.ravel(), yy.ravel())))  # Assign grid points to input node
y_node.value = np.zeros((xx.ravel().shape[0], 1))  # Dummy labels for grid points
forward_pass(graph)  # Perform forward pass

Z = np.array(sigmoid.value).reshape(xx.shape)  # Reshape predictions to grid

plt.contourf(xx, yy, Z, alpha=0.8)  # Plot decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)  # Plot training data
plt.xlabel('Feature 1')
plt.colorbar()
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

plt.plot(epoch_array, loss_array)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
