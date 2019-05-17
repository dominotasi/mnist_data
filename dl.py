import mnist
import numpy as np

class NeuralNetwork(object):
    def __init__(self, layer_nodes, filter_list):
        assert (len(layer_nodes) == len(filter_list) + 1)
        self.layers = layer_nodes
        self.functions = filter_list
        self.weights = []
        self.biases = []

        for i in range(len(layer_nodes)-1):
            self.weight = np.random.normal(0, 1, [layer_nodes[i+1] , layer_nodes[i]])
            self.bias = np.zeros((layer_nodes[i + 1],1))
            self.weights.append(self.weight)
            self.biases.append(self.bias)


def forward(x, W, b):
    # x: input sample (N, 2)
    # W: Weight (2,)
    # b: bias float
    # out: (N,)
    out = x.dot(W) + b        # Multiple X with W + b: (N, 2) * (2,) -> (N,)
    cache = (x, W, b)
    return out, cache

def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    N = X.shape[0]            # Find the number of samples
    loss = np.sum(np.square(h - y)) / N   # Compute the mean square error from its true value y
    dout = 2 * (h-y) / N                  # Compute the partial derivative of J relative to out
    return loss, dout

def backward(dout, cache):
    # dout: dJ/dout (N,)
    # x: input sample (N, 2)
    # W: Weight (2,)
    # b: bias float
    x, W, b = cache
    dw = x.T.dot(dout)            # Transpose x (N, 2) -> (2, N) and multiple with dout. (2, N) * (N, ) -> (2,)
    db = np.sum(dout, axis=0)     # Add all dout (N,) -> scalar
    return dw, db

def compute_loss(X, W, b, y=None):
    h, cache = forward(X, W, b)

    if y is None:
        return h

    loss, dout = mean_square_loss(h, y)
    dW, db = backward(dout, cache)
    return loss, dW, db



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # load data
    num_classes = 10
    train_images = mnist.train_images()  # [60000, 28, 28]
    # print(train_images)
    train_labels = mnist.train_labels()
    # print("train label: " + str(train_labels))
    test_images = mnist.test_images()
    # print("test images: " + str(test_images))
    test_labels = mnist.test_labels()
    # print("test lbl: " + str(test_labels))

    # data processing
    X_train = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]).astype(
        'float32')  # flatten 28x28 to 784x1 vectors, [60000, 784]
    # print("x shape: " + str(X_train.shape))
    x_train = X_train / 255  # normalization
    print("x_train shape: " + str(x_train.shape))
    y_train = np.eye(num_classes)[train_labels]  # convert label to one-hot
    # print("y train: " + str(y_train.shape))

    X = x_train
    y = y_train
    a = 0
    for i in range(iteration):
        loss, dW, db = compute_loss(X, weights[a], biases[a], y)
        weights[a] -= learning_rate * dW
        biases[a] -= learning_rate * db
        if i%100==0:
            print(f"iteration {i}: loss={loss:.4} W1={W[0]:.4} dW1={dW[0]:.4} W2={W[1]:.4} dW2={dW[1]:.4} b= {b:.4} db = {db:.4}")

    print(f"W = {W}")
    print(f"b = {b}")