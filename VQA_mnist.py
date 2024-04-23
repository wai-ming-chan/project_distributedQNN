import numpy as np
from keras.datasets import mnist
import cv2 # OpenCV for image interpolation

# set random seed for reproducibility
np.random.seed(0)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('---------------------------------')
print('Original Dataset Shapes:')
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

input_size = train_X.shape[1] * train_X.shape[2]
output_size = np.unique(train_y).shape[0]

print(f'input size: log2(28*28) = {np.log2(input_size)}')
print(f'output size: {np.log2(output_size)}')

# filter to smaller images (28*28) -> (6*6)
# Resize the all images to 8x8 using bilinear interpolation
# e.g.: smaller_image = cv2.resize(mnist_image, (6, 6), interpolation=cv2.INTER_LINEAR)
new_image_size = (8, 8)
# Function to resize images to 6x6
def resize_images(images):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, new_image_size, interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized_img)
    return np.array(resized_images)

# Resize training and test images to new_image_size
train_X_resized = resize_images(train_X)
test_X_resized = resize_images(test_X)

# num_classes = len(np.unique(train_y)) # this value should be 10
num_classes = 4 # use a smaller number of classes for small model training
# Number of training and test samples per class
num_train_per_class = 40
num_test_per_class = 20
num_train = num_train_per_class * num_classes
num_test = num_test_per_class * num_classes


train_X_resized = train_X_resized[train_y < num_classes]
train_y = train_y[train_y < num_classes]
test_X_resized = test_X_resized[test_y < num_classes]
test_y = test_y[test_y < num_classes]



# select the subset of training and test samples per class
train_X_resized_subset = []
train_y_subset = []
test_X_resized_subset = []
test_y_subset = []

for i in range(num_classes):
    train_X_resized_subset.append(train_X_resized[train_y == i][:num_train_per_class])
    train_y_subset.append(train_y[train_y == i][:num_train_per_class])
    test_X_resized_subset.append(test_X_resized[test_y == i][:num_test_per_class])
    test_y_subset.append(test_y[test_y == i][:num_test_per_class])

train_X_resized = np.concatenate(train_X_resized_subset)
train_y = np.concatenate(train_y_subset)
test_X_resized = np.concatenate(test_X_resized_subset)
test_y = np.concatenate(test_y_subset)

# Shuffle the training and test sets
shuffle_train = np.random.permutation(len(train_X_resized))
shuffle_test = np.random.permutation(len(test_X_resized))   

train_X_resized = train_X_resized[shuffle_train]
train_y = train_y[shuffle_train]
test_X_resized = test_X_resized[shuffle_test]
test_y = test_y[shuffle_test]


X_train = train_X_resized.copy()
X_test = test_X_resized.copy()
Y_train = train_y.copy()
Y_test = test_y.copy()

# Print shapes of resized training and test sets
print('---------------------------------')
print('Resized Dataset Shapes:')
print('X_train_resized: ' + str(X_train.shape))
print('X_test_resized:  '  + str(X_test.shape))

# display the original (4 images, distinct labels)
import matplotlib.pyplot as plt
(train_X, train_y), (_, _) = mnist.load_data()
fig, axes = plt.subplots(1, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    # get image for class i
    img = train_X[train_y == i][0]

    ax.imshow(img, cmap='gray')    
    ax.set_title(f'Label: {i}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('original_mnist.png')
plt.show()

# dispay the resized images (4 images, distinct labels)
fig, axes = plt.subplots(1, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    # get image for class i
    img = X_train[Y_train == i][0]

    ax.imshow(img, cmap='gray')    
    ax.set_title(f'Label: {i}')
    ax.axis('off')
plt.tight_layout()

plt.savefig('resized_mnist.png')
plt.show()

# Normalize the resized images to the range [0, 1]
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# convert the images to vectors
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
####
####
####
####
#### Finish data preprocessing
##################################################
####
####
# Train a simple fully connected neural network (using Pytorch) to classify the resized images (8x8)
# The network consists of 2 layers: input layer (64 neurons) and output layer (10 neurons)
# The output layer employs a softmax activation function
# The loss function employs cross entropy loss
# The optimizer employs Adam
# The learning rate is set to 0.001

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

# Create DataLoader objects for training and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple fully connected neural network (no hard-coded values)
# one layer only (input size * output size)
class OneLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneLayerNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # output (relu + softmax)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x
    
# Create an instance of the OneLayerNN class
input_size = new_image_size[0] * new_image_size[1]
output_size = num_classes
model = OneLayerNN(input_size, output_size)

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
batch_size = 40
# keep track of the loss and accuracy for each epoch
train_loss, test_loss = [], []
train_accuracy, test_accuracy = [], []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    # Train the model using minibatches
    for it in range(0, len(X_train), batch_size):
        inputs = X_train_tensor[it:it+batch_size]
        labels = y_train_tensor[it:it+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = F.one_hot(labels, num_classes).float() # convert labels to matrix form

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Compute accuracy and loss for the training set after each epoch
    with torch.no_grad():
        # Training set
        outputs = model(X_train_tensor)

        loss = criterion(outputs,  F.one_hot(y_train_tensor, num_classes).float() )
        running_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total = y_train_tensor.size(0)
        correct = (predicted == y_train_tensor).sum().item()
        epoch_loss = running_loss 
        epoch_accuracy = correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Testing set
        outputs = model(X_test_tensor)
        loss = criterion(outputs,  F.one_hot(y_test_tensor, num_classes).float())
        running_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total = y_test_tensor.size(0)
        correct = (predicted == y_test_tensor).sum().item()
        epoch_loss = running_loss 
        epoch_accuracy = correct / total
        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss[-1]:.4f}/{test_loss[-1]:.4f}, Accuracy: {train_accuracy[-1]:.4f}/{test_accuracy[-1]:.4f}')
        ####
        ####
        ####
# Print the total number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f'[Classical FNN] Total number of parameters: {num_params}')

# Plot the loss and accuracy for training and test sets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classical One-Layer NN Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Classical One-Layer NN Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('FNN_loss_plot.png')
plt.show()

##################################################
#
# Part 2: Quantum Neural Network (QNN) for VQA on MNIST Dataset
# model architecture: 4 qubits,...
import pennylane as qml
from pennylane import numpy as jnp
from pennylane.optimize import AdamOptimizer
# import jax
# import jax.numpy as jnp

# Our circuit layer will use 12 qubits, or wires, and consists of an arbitrary rotation on every qubit, as well as a ring of CNOTs that entangles each qubit with its neighbour. Borrowing from machine learning, we call the parameters of the layer weights.
num_wires = 2

# CNOT_connection_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
CNOT_chain_range = jnp.arange(num_wires)
CNOT_connection_pairs = [[i, i+1] for i in range(num_wires-1)]

num_meas_qubits = int(np.ceil(jnp.log2(num_classes))) # number of measurement qubits = log2(num_classes)
print(f'num_meas_qubits: {num_meas_qubits}')

# We create a quantum device that will run our circuits.
# dev = qml.device("lightning.gpu", wires=num_wires) # error (not working in my computer)
dev = qml.device("default.qubit", wires=num_wires)
# dev = qml.device('default.qubit.jax', wires=2) # use jax backend


def layer(layer_weights): #, f=None):
    for wire in range(num_wires):
        qml.Rot(*layer_weights[wire], wires=wire)

    # qml.broadcast(unitary=qml.CNOT, pattern='chain', wires=CNOT_chain_range, parameters=None)
    
    for wires in CNOT_connection_pairs:
        qml.CNOT(wires)
        

# Now we define the variational quantum circuit as this state preparation routine, followed by a repetition of the layer structure.
# @jax.jit  # QNode calls will now be jitted, and should run faster.
# @qml.qnode(dev, interface='jax')
@qml.qnode(dev)
def circuit(weights, x):
    # data reuploading
    for i in range(len(weights)):
        inner_weights = weights[i]
        
        # qml.AmplitudeEmbedding(features=x, wires=range(num_wires), pad_with=0.)
        # use phase encoding
        # determine number of phase (Rot) encoding layers for fixed number of qubits, so that one layer can encode num_wires * 3 features
        num_phase_encoding_layers = int(jnp.ceil(len(x) / num_wires / 3))
        data_index = 0 # keep track of the index of the data for encoding, encode 3 features at a time
        for _ in range(num_phase_encoding_layers):
            for j in range(num_wires):
                if data_index + 3 < len(x):
                    # qml.RX(x[data_index], wires=j)
                    # data_index += 1
                    # qml.RY(x[data_index], wires=j)
                    # data_index += 1
                    # qml.RZ(x[data_index], wires=j)
                    # data_index += 1
                    qml.Rot(x[data_index], x[data_index+1], x[data_index+2], wires=j)
                    data_index += 3
                else:
                    break

            for layer_weights in inner_weights:
                layer(layer_weights)

        
    return qml.probs(wires=range(num_wires-num_meas_qubits, num_wires)) # measure the last "num_meas_qubits" qubits
    # return qml.probs(wires=[8,9,10,11])
    # return qml.expval(qml.PauliZ(0))

# If we want to add a “classical” bias parameter, the variational quantum classifier also needs some post-processing. We define the full model as a sum of the output of the quantum circuit, plus the trainable bias.
def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

# In supervised learning, the cost function is usually the sum of a loss function and a regularizer. We restrict ourselves to the standard square loss that measures the distance between target labels and model predictions.
def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return jnp.mean((labels - qml.math.stack(predictions)) ** 2)

# To monitor how many inputs the current classifier predicted correctly, we also define the accuracy, or the proportion of predictions that agree with a set of target labels.
def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

# For learning tasks, the cost depends on the data - here the features and labels considered in the iteration of the optimization routine.
def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions) 

FLAG_DRAW_CIRCUIT = False
if FLAG_DRAW_CIRCUIT:
    weights_init = 0.01 * jnp.random.randn(num_layers, num_inner_layers, num_wires, 3, requires_grad=True)
    x = X_train[0]
    fig, ax = qml.draw_mpl(circuit)(weights_init, x)
    fig.savefig('VQA_circuit.png')
    fig.show()

#---------------------------------------------------
# Optimization (Train the VQA model)
#---------------------------------------------------
# We initialize the variables randomly (but fix a seed for reproducibility). Remember that one of the variables is used as a bias, while the rest is fed into the gates of the variational circuit.
# jnp.random.seed(0) # 'jax.numpy' has no attribute 'random'

num_layers = 6  # number of layers for the outer loop
# num_layers = int(np.ceil(num_params /num_wires / 3  )) # number of layers for the outer loop
print(f'VQA num_layers: {num_layers}')
num_inner_layers = 1 # number of layers for the inner loop

weights_init = 0.01 * jnp.random.randn(num_layers, num_inner_layers, num_wires, 3, requires_grad=True)
bias_init = jnp.array(jnp.zeros(2**num_wires), requires_grad=True)

print(f'size of Weight: {weights_init.size} | size of bias: {bias_init.size} | total: {weights_init.size + bias_init.size}')

# Next we create an optimizer instance and choose a batch size…
# opt = NesterovMomentumOptimizer(0.5)
opt = AdamOptimizer(0.001, beta1=0.9, beta2=0.999)


# run the optimizer to train our model
weights = weights_init
bias = bias_init
X_train_VQA = X_train.copy()
X_test_VQA = X_test.copy()
Y_train_VQA = Y_train.copy()
Y_test_VQA = Y_test.copy()

# Normalize the data to the range [-pi, pi]
X_train_VQA = (X_train_VQA - 0.5) * 2 * jnp.pi
X_test_VQA = (X_test_VQA - 0.5) * 2 * jnp.pi

# Extend the output labels to size 2^num_meas_qubits
Y_train_VQA = [np.eye(2**num_meas_qubits)[Y_train_VQA[i]] for i in range(len(Y_train_VQA))]
Y_test_VQA = [np.eye(2**num_meas_qubits)[Y_test_VQA[i]] for i in range(len(Y_test_VQA))]
print(f'Y_train_VQA: { np.shape(Y_train_VQA)} | Y_test_VQA: {np.shape(Y_test_VQA)}')

# keep track of the loss and accuracy for each epoch
train_loss_VQA, test_loss_VQA = [], []
train_accuracy_VQA, test_accuracy_VQA = [], []
for epoch in range(num_epochs):
    # Train the model using minibatches
    # shuffle the training set
    shuffle_train = np.random.permutation(len(X_train_VQA))
    for it in range(0, len(X_train_VQA), batch_size):

        # X_batch = X_train_VQA[it:it+batch_size]
        X_batch = X_train_VQA[shuffle_train[it:it+batch_size]]
        Y_batch = [Y_train_VQA[shuffle_train[i]] for i in range(it, it+batch_size)]

        weights, bias = opt.step(lambda w, b: cost(w, b, X_batch, Y_batch), weights, bias)

    # Compute accuracy and loss after each epoch
    # Training set
    current_cost = cost(weights, bias, X_train_VQA, Y_train_VQA)
    predictions_train = [variational_classifier(weights, bias, x) for x in X_train_VQA]
    predictions_train_idx = [jnp.argmax(predictions_train[i]) for i in range(len(predictions_train))]
    Y_train_idx = [jnp.argmax(Y_train_VQA[i]) for i in range(len(Y_train_VQA))]
    acc_train = accuracy(Y_train_idx, predictions_train_idx)
    train_accuracy_VQA.append(acc_train)
    train_loss_VQA.append(current_cost)
    
    # Testing set
    current_cost = cost(weights, bias, X_test_VQA, Y_test_VQA)
    predictions_test = [variational_classifier(weights, bias, x) for x in X_test_VQA]
    predictions_test_idx = [jnp.argmax(predictions_test[i]) for i in range(len(predictions_test))]
    Y_test_idx = [jnp.argmax(Y_test_VQA[i]) for i in range(len(Y_test_VQA))]
    acc_test = accuracy(Y_test_idx, predictions_test_idx)
    test_accuracy_VQA.append(acc_test)
    test_loss_VQA.append(current_cost)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss_VQA[-1]:.4f}/{test_loss_VQA[-1]:.4f}, Accuracy: {train_accuracy_VQA[-1]:.4f}/{test_accuracy_VQA[-1]:.4f}')
# finish training