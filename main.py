import mnist
import numpy as np
import train as nt
import functions as f

# load data
num_classes = 10
train_images = mnist.train_images() #[60000, 28, 28]
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# data processing
X_train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_train = X_train / 255 #normalization
y_train = np.eye(num_classes)[train_labels] #convert label to one-hot

X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_test = X_test / 255 #normalization
y_test = test_labels

#f.plot_digit(train_images, train_labels, 1)

#func_list = ['sigmoid','sigmoid', 'relu', 'stable_softmax']  # kullanmak istediğimiz filtreler sırasıyla
func_list = ['relu', 'sigmoid', 'softmax']

layer = nt.Layer(
    inputs= x_train,
    labels= y_train,
    layer_nodes=[784, 20, 10],
    batch_size= 1,
    epochs=1,
    learning_rate= 0.001

)


#layer.network(inputs= x_train, labels= y_train,func_list=func_list, layer_count=4, layer_nodes=[784, 200, 100, 10])
layer.train(inputs= x_train, labels= y_train, func_list=func_list)
