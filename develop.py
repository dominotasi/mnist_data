
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:22:04 2019

@author: domin
"""


import numpy as np
#import mnist_data as md
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
            self.weight = np.random.normal(0, 1, [layer_nodes[i+1] , layer_nodes[i]])
            self.bias = np.zeros((layer_nodes[i + 1],1))
            #print("weight: " + str(self.weight))
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
                print("labels_batch:" + str(labels_batch))
                z = []
                activations = []
                i = 0
                for func in func_list:
                    if self.index == 0:
                        self.z = np.dot(self.weights[self.index], inputs_batch.T) + self.biases[self.index]
                        print("merve")
                    else:
                        self.z = np.dot(self.weights[self.index], activations[self.index-1]) + self.biases[self.index]
                        i += 1
                        print("domino")
                    z.append(self.z)
                    #print("Z: " + str(self.z))
                    print("z's shape: " + str(self.z.shape))
                    self.y = self.functions[func](self.z)
                    activations.append(self.y)
                    #print("act shape: " + str(activations[i]))
                    print(i)
                    self.index += 1
                    #print("y' : " + str(self.activation))
                    #print("y' 's shape: " + str(self.y.shape))
                    print("index = " +str(self.index))
                    #print(self.functions[func])
                self.dongu +=1
                print("döngü= " + str(self.dongu))
                # print(activation.shape)
                #print("hellö")
                print("acti: " + str(activations[self.index-1]))
                # self.index = 0
                a = 0
                print("merveeee")
                loss = functions.cross_entropy_loss(activations[self.index-1], labels_batch)
                #loss = functions.mean_square_loss(activations[self.index-1], labels_batch)
                loss += functions.L2_regularization(0.01, self.weights[a],self.weights[a+1])  # lambda
                self.loss.append(loss)
                self.index = 0
               #print('<<<<==== Epoch: {:d}/{:d} -- Iteration:{:d} -- Loss: {:.2f} ====>>>>'.format(epoch+1, self.epochs, iteration+1, loss))
                iteration += self.batch_size

