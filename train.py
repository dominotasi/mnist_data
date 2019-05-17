# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:22:04 2019

@author: domin
"""


import numpy as np
import mnist_data as md
import functions

#import mnist


class Layer(object):
    def __init__(self, inputs, labels, batch_size, epochs, learning_rate, layer_nodes):
        self.loss = []
        self.weight = []
        self.bias = []
        self.index = 0
        self.inputs = inputs
        self.labels = labels
        self.layer_nodes = layer_nodes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.y = None
        self.epochs = epochs
        self.dongu = 0
        self.functions = {
            'softmax': functions.softmax,
            'stable_softmax': functions.stable_softmax,
            'sigmoid': functions.sigmoid,
            'relu': functions.relu


        }

        self.weights = []
        self.biases = []
        # weight bias oluşturma
        for i in range(len(layer_nodes)-1):
            self.weight = np.random.normal(0, 1, [layer_nodes[i+1], layer_nodes[i]])
            self.bias = np.zeros((layer_nodes[i + 1], 1))
            # print("weight: " + str(self.weight))
            print("weight's shape: " + str(self.weight.shape))
            # print("bias: " + str(self.bias))
            print("bias's shape: " + str(self.bias.shape))
            # print("bias: " + str(bias))
            self.weights.append(self.weight)
            self.biases.append(self.bias)



    def train(self, inputs, labels, func_list):
        print("train")
        for epoch in range(0, self.epochs):  # training başlangıcı
            iteration = 0
            while iteration < len(inputs):
                inputs_batch = inputs[iteration:iteration + self.batch_size]
                #print("input batch: " + str(inputs_batch))
                labels_batch = labels[iteration:iteration + self.batch_size]
                #print("labels_batch:" + str(labels_batch))
                a = 0
                z = []
                activations = []
                for func in func_list:
                    if self.index == 0:
                        self.z = np.dot(inputs_batch, self.weights[self.index]) + self.biases[self.index]
                    else:
                        self.z = np.dot(self.z, self.weights[self.index]) + self.biases[self.index]
                    z.append(self.z)
                    #print("Z: " + str(self.z))
                    #print("z's shape: " + str(self.z.shape))
                    self.y = self.functions[func](self.z)
                    activations.append(self.y)
                    self.index += 1
                    #print("y' : " + str(self.activation))
                    #print("y' 's shape: " + str(self.y.shape))
                    #print("index = " +str(self.index))
                    #print(self.functions[func])
                self.dongu +=1
                #print("döngü= " + str(self.dongu))
                # print(activation.shape)
                #print("hellö")

                self.index = 0

                loss = functions.cross_entropy_loss(self.y, labels_batch)
                loss += functions.L2_regularization(0.01, self.weights[a],self.weights[a+1])  # lambda
                self.loss.append(loss)
                #print("loss: " + str(self.loss))

                i = 0
                delta_y = (self.y - labels_batch) / self.y.shape[0]
                #print("delta y" + str(delta_y.shape))
                # print("delta_y:" + str(delta_y))
                #print("len:" + str(len(self.weights)))
                m = len(self.weights)
                #print(m)
                merve = 0
                delta_hl = []
                for a in range(m-1):
                    delta_h = np.dot(delta_y, self.weights[m-1].T)
                    m -= 1
                    merve += 1
                    delta_hl.append(delta_h)
                    #print("delta_h:" + str(delta_h.shape))
                # burada relu varsa bunu yap yaz
                #if 'relu' in func_list:
                 #   delta_h[activations[i] <= 0] = 0 # derivative relu garanti olsun
                # print("delta_h:" + str(delta_h))
                #m -= 1
                #if m-1 != 0:
                #    delta_h = np.dot(delta_y, weights[m - 1].T)
                #    delta_h[activations[i] <= 0] = 0
                #else:

                # backpropagation
                k = len(self.weights)
                if len(self.layer_nodes)-1 == k:
                    weight_gradient = np.dot(activations[m-1].T, delta_y)  # forward * backward  m-1 çünkü m=len(weights)
                    #print(weight_gradient)
                    bias_gradient = np.sum(delta_y, axis=0, keepdims=True)
                    #print("weights gradient: " + str(weight_gradient.shape))
                    #print("weights (k-1): " + str(self.weights[k-1].shape))
                    weight_gradient += 0.01 * self.weights[k-2]
                    self.weights[k-2] -= self.learning_rate * weight_gradient  # weight ve bias güncelleme
                    self.biases[k-2] -= self.learning_rate * bias_gradient
                    k -= 1
                    # burayı halledersek tamamdır
                elif k == 1:
                    weight_gradient = np.dot(inputs_batch.T, delta_hl[merve])
                    bias_gradient = np.sum(delta_hl[merve], axis=0, keepdims = True)
                    weight_gradient += 0.01 * self.weights[k - 1]
                    self.weights[k - 1] -= self.learning_rate * weight_gradient
                    self.bias[k - 1] -= self.learning_rate * bias_gradient
                    merve -= 1
                else:
                    weight_gradient = np.dot(delta_hl[merve].T, delta_hl[merve-1])
                    bias_gradient = np.sum(delta_hl[merve-1], axis=0, keepdims = True)
                    weight_gradient += 0.01 * self.weights[k - 1]
                    self.weights[k - 1] -= self.learning_rate * weight_gradient
                    self.biases[k - 1] -= self.learning_rate * bias_gradient
                    merve -=1
                print('<----=== Epoch: {:d}/{:d} -- Iteration:{:d} -- Loss: {:.2f} ===---->'.format(epoch+1, self.epochs, iteration+1, loss))
                iteration += self.batch_size