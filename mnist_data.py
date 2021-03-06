from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt

X, y = loadlocal_mnist(
    images_path="C:/Users/domin/PycharmProjects/Mnist_Data/daten/train-images.idx3-ubyte",
    labels_path="C:/Users/domin/PycharmProjects/Mnist_Data/daten/train-labels.idx1-ubyte")

X_test, y_test = loadlocal_mnist(
    images_path="C:/Users/domin/PycharmProjects/Mnist_Data/daten/t10k-images.idx3-ubyte",
    labels_path="C:/Users/domin/PycharmProjects/Mnist_Data/daten/t10k-labels.idx1-ubyte"
)
"""
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])



print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))

"""
# np.savetxt(fname='C:/Users/domin/PycharmProjects/Mnist_Data/images.csv',
#           X=X, delimiter=',', fmt='%d')
# np.savetxt(fname='C:/Users/domin/PycharmProjects/Mnist_Data/labels.csv',
 #          X=y, delimiter=',', fmt='%d')
# np.savetxt(fname='C:/Users/domin/PycharmProjects/Mnist_Data/testimages.csv',
 #          X=X_test, delimiter=',', fmt='%d')
# np.savetxt(fname='C:/Users/domin/PycharmProjects/Mnist_Data/testlabels.csv',
 #          X=y_test, delimiter=',', fmt='%d')

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')  #beyaz üzerine siyah---gray ise siyah üzerine beyaz
    plt.title('true label: %d' % y[idx])
    plt.show()

#plot_digit(X, y, 0)
#print(X)