# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:22:04 2019

@author: domin
"""


import numpy as np
import mnist_data as md

#import mnist


class Layer(object):
    def __init__(self, layer_count, inputs, labels, batch_size):
        self.loss = []
        self.weight = None
        self.bias = None
        self.index = 0
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.activation = None
        self.layer_count = layer_count
        self.dongu = 0
        self.functions = {
            'softmax': self.softmax,
            'stable_softmax': self.stable_softmax,
            'sigmoid': self.sigmoid,
            'relu': self.relu,
            'domino': self.domino,

        }

    def domino(self, z):
        merve = z
        return merve

    def softmax(self, z):
        exps = np.exp(z)
        return exps / np.sum(exps)

    def stable_softmax(self, z):
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps)

    def sigmoid(self, z):
        #sigma = [1 / float(1 + np.exp(-x)) for x in z]
        sigma = 1.0/(1.0+np.exp(-z))
        return sigma

    def tanh(self, z):
        return np.tanh(z)


    def relu(self, z):
        return np.maximum(0.0, z)




    def network(self,inputs, labels, func_list, layer_count, layer_nodes):
        iteration = 0
        inputs_batch = inputs[iteration:iteration + self.batch_size]
        # print("input batch: " + str(inputs_batch))
        labels_batch = labels[iteration:iteration + self.batch_size]
        # print("labels_batch:" + str(labels_batch))

        weights = []
        bias = []
        z = []
        activations = []
        for func in func_list:
            self.weight = np.random.random((layer_nodes[self.index], layer_nodes[self.index + 1]))
            self.bias = np.zeros((1, layer_nodes[self.index + 1]))
            # print("weight: " + str(self.weight))
            # print("weight's shape: " + str(self.weight.shape))
            weights.append(self.weight)
            bias.append(self.bias)
            # print("bias: " + str(self.bias))
            # print("bias's shape: " + str(self.bias.shape))
            print("bias: " + str(bias))
            if self.index == 0:
                self.z = np.dot(inputs_batch, self.weight) + self.bias
            else:
                self.z = np.dot(self.z, self.weight) + self.bias
            z.append(self.z)
            # print("Z: " + str(self.z))
            print("z's shape: " + str(self.z.shape))
            self.y = self.functions[func](self.z)
            activations.append(self.y)
            # print("y' : " + str(self.activation))
            print("y' 's shape: " + str(self.y.shape))
            self.index += 1
            print("index = " + str(self.index))
            print(self.functions[func])
        self.dongu += 1
        print("döngü= " + str(self.dongu))
        # print(activation.shape)
        print("hellö")


        """
        for func in func_list:
            if self.index == layer_count-1:
                self.weight = np.random.random((layer_nodes[self.index], 10))
                self.bias = np.zeros((1, 10))
            else:
                self.weight = np.random.random((layer_nodes[self.index], layer_nodes[self.index + 1]))
                self.bias = np.zeros((1, layer_nodes[self.index + 1]))
            print("weight: " + str(self.weight))
            print("weight's shape: " + str(self.weight.shape))
            print("bias: " + str(self.bias))
            print("bias's shape: " + str(self.bias.shape))
            if self.index == 0:
                self.z = np.dot(self.inputs, self.weight) + self.bias
            #elif self.index == layer_count-1:
            #    self.z = np.dot()
            else:
                self.z = np.dot(self.z, self.weight) + self.bias

            print("Z: " + str(self.z))
            print("z's shape: " + str(self.z.shape))
            self.activation = self.functions[func](self.z)
            print("Activation: " + str(self.activation))
            print("activation's shape: " + str(self.activation.shape))
            self.index += 1
            print("index = " +str(self.index))
            print(self.functions[func])
        self.dongu +=1
        print("döngü= " + str(self.dongu))
        # print(activation.shape)
        print("hellö")
        """


if __name__ == "__main__":

    func_list = ['sigmoid', 'sigmoid', 'relu', 'stable_softmax']      #kullanmak istediğimiz filtreler sırasıyla

    x_train = md.X[0]
    x_labels = md.y


    layer = Layer(
        layer_count=4,
        inputs=x_train

    )

    layer.network( func_list=func_list, layer_count=4, layer_nodes=[784, 200, 100, 10])
