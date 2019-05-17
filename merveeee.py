import numpy as np
import mnist

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

    def feedforward(self, inputs_batch,filter_list):
        # return the feedforward value for x
        index = 0
        merve = 0
        z_s = []
        z = []
        activations = []
        i = 0
        for i in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.functions[i])
            if index == 0:
                z = np.dot(self.weights[index], inputs_batch.T) + self.biases[index]
                print("merve")
            else:
                z = np.dot(self.weights[index], activations[index - 1]) + self.biases[index]
                print("domino")
            z_s.append(z)
            # print("Z: " + str(self.z))
            print("z's shape: " + str(z.shape))
            y_hat = activation_function(z_s[i])
            activations.append(y_hat)
            print("act shape: " + str(activations[i]))
            index += 1
            # print("y' : " + str(self.activation))
            # print("y' 's shape: " + str(self.y.shape))
            print("index = " + str(index))
            # print(self.functions[func])
        return (z_s, activations)

    def bachward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        delta_weight = np.dot(input.T, grad_output)
        delta_bias = grad_output.mean(axis=0) * input.shape[0]

        assert delta_weight.shape == self.weights.shape and delta_bias.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        #self.weights = self.weights - self.learning_rate * grad_weights
        #self.biases = self.biases - self.learning_rate * grad_biases

        return delta_weight, delta_bias


    def backpropagation(self, y_hat, z_s, activations):
        delta_weight = []  # dC/dW
        delta_bias = []  # dC/dB
        deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
        deltas[-1] = ((y_hat - activations[-1]) * (self.getDerivitiveActivationFunction(self.functions[-1]))(z_s[-1]))
        # print("deltas 1 :" + str(deltas[-1]))
        print("deltas 1 shape :" + str(deltas[-1].shape))

        # Perform BackPropagation
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(self.weights[i + 1].T,deltas[i + 1]) * (self.getDerivitiveActivationFunction(self.functions[i])(z_s[i]))
            print("hellö")
            # print("delta: " + str(deltas[i]))
            print("delta shape: " + str(deltas[i].shape))

            #a= [print(d.shape) for d in deltas]
        batch_size = y_hat.shape[1]
        delta_bias = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
        # print("delta_bias: " + str(delta_bias))
        print("delta_bias len: " + str(len(delta_bias)))

        delta_weight = [d.dot(activations[i].T) / float(batch_size) for i, d in enumerate(deltas)]
        # return the derivatives respect to weight matrix and biases
        return delta_weight, delta_bias



    @staticmethod
    def getActivationFunction(name):
        if (name == 'sigmoid'):
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))

            return sigmoid
        elif (name == 'tanh'):
            def tanh(x):
                return np.tanh(x)
            return tanh

        elif (name == 'relu'):
            def relu(x):
                y = np.copy(x)
                y[y < 0] = 0
                return y

            return relu
        elif (name == 'softmax'):
            def softmax(z):
                exp = np.exp(z)
                return exp / np.sum(exp, axis=1, keepdims=True)
            return softmax
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x

    @staticmethod
    def getDerivitiveActivationFunction(name):
        if (name == 'sigmoid'):
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))

        elif (name == 'relu'):
            def relu_derivative(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y

            return relu_derivative
        elif (name == 'tanh'):
            def tanh(x):
                return np.tanh(x)

            return tanh

        elif (name == 'softmax'):
            def softmax(z):
                exp = np.exp(z)
                return exp / np.sum(exp, axis=1, keepdims=True)

            return softmax
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1

    def train(self, inputs, labels, batch_size, epochs, learning_rate):
        # update weights and biases based on the output
        for epoch in range(epochs):  # training başlangıcı
            iteration = 0
            while iteration < len(inputs):
                inputs_batch = inputs[iteration:iteration + batch_size]
                labels_batch = labels[iteration:iteration + batch_size]
                iteration = iteration + batch_size
                z_s, activations = self.feedforward(inputs_batch, filter_list=['relu', 'sigmoid', 'softmax'])
                dw, db = self.backpropagation(labels_batch, z_s, activations)
                self.weights = [w + learning_rate * dweight for w, dweight in zip(self.weights, dw)]
                self.biases = [w + learning_rate * dbias for w, dbias in zip(self.biases, db)]
                print("=== Epoch: {:d}/{:d} -- Iteration:{:d} -- Loss = {}".format(epoch+1, epochs, iteration+1, (np.linalg.norm(activations[-1] - labels_batch))))


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

    nn = NeuralNetwork(layer_nodes=[784, 200, 100, 10], filter_list= ['relu', 'sigmoid', 'softmax'])
    inputs= x_train,
    labels= y_train,


    nn.train(inputs = x_train, labels= y_train, epochs=10000, batch_size=1, learning_rate=.1)
    z_s, activations = nn.feedforward(inputs)
    # print(y, X)
