import numpy as np
import matplotlib.pyplot as plt
import mnist_data as md

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')  #beyaz üzerine siyah---gray ise siyah üzerine beyaz
    plt.title('true label: %d' % y[idx])
    plt.show()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(z, 0.0)

def tanh(z):
    return np.tanh(z)

def softmax(z):
    exp = np.exp(z)
    return exp/np.sum(exp, axis = 1, keepdims = True)


def mean_square_loss(x, y):
    N = md.X[0].shape
    print(N)
    loss = np.sum(np.square(x - y)) / N
    out = 2 * (x -y) / N
    return loss, out

def cross_entropy_loss(x,y):
    # m = y.shape[0]
    #print(m)
    #p = softmax(x)
    #print("p: " + str(p))
    #log_likelihood = -np.log(p[range(m), y], dtype= float)
    #print(log_likelihood)
    #loss = np.sum(log_likelihood) / m
    #return loss
    # index = y.shape[0]
    merve = np.argmax(y, axis = 1).astype(int)
    print(merve)
    print(y)
    probability = x[np.arange(len(x)), merve]
    #probability = np.log(x[range(len(x)),y])
    log = np.log(probability)
    loss = -1.0 * np.sum(log) / len(log)
    return loss


def L2_regularization(lam, vorweight, nachweight):
    vorweight_loss = 0.5 * lam * np.sum(vorweight * vorweight)
    nachweight_loss = 0.5 * lam * np.sum(nachweight * nachweight)
    return vorweight_loss + nachweight_loss


def stable_softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

def mean_square_error(x, y):
    N = md.X[0].shape[0]
    loss = np.sum(np.square(x - y)) / N
    return loss



x = np.arange(-20, 20, 0.001)
print(x)
plt.plot(x)
# plt.show()

y = sigmoid(x)
plt.axvline(x=0, color="0.8")
plt.plot(x,y)
# plt.show()

z = mean_square_loss(x, y)
print(z)


"""
print(x.shape)
z = relu(x)
print(z)
print(z.shape)
plt.axvline(x=0, color="0.8")
plt.axhline(y=0, color="0.8")
plt.plot(x,z)
plt.show()

k = tanh(x)
plt.plot(x,k)
plt.axhline(y=0, color="0.8")
plt.show()

m = softmax(y)
plt.plot(y,m)
plt.show()

"""
