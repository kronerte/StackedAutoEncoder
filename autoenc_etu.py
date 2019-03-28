#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:46:02 2018

@author: johnklein
"""


import tensorflow as tf
# if the import fails, try to install tf : pip install --upgrade tensorflow
from sklearn.preprocessing import StandardScaler
import numpy.random as rnd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# to work on the same dataset as students
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    
####################
#Dataset generation#
####################

rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(X_train[:,0], X_train[:,1], X_train[:,2],'.', label='dataset')
ax.legend()
plt.savefig("dataset")
plt.show()

##########################
#PCA with an auto-encoder#
##########################

d_in = 3 #input dimensionality
d_hid = 2 #code dimensionality
d_out = d_in #output dimensionality

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()
activation = tf.nn.elu

#Input data
X = tf.placeholder(tf.float32,shape=[None, d_in])

#Hidden layer (code generating layer)
weights1_init = initializer([d_in, d_hid])
weights1 = tf.Variable(weights1_init,dtype=tf.float32,name="weights1")
biases1 = tf.Variable(tf.zeros(d_hid),name="biases1")
hidden = activation(tf.matmul(X,weights1) + biases1)

#Output layer (input reconstruction)
weights2 = tf.transpose(weights1,name="weights2")
biases2 = tf.Variable(tf.zeros(d_out),name="biases2")
outputs = activation(tf.matmul(hidden,weights2) + biases2)

#Objective function: MSE
J = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(J)

init = tf.global_variables_initializer()

n_iterations = 1000
error_hist = []
with tf.Session() as sess:
    init.run()
    reduced_dataset = sess.run(hidden, feed_dict={X: X_train})
    plt.scatter(reduced_dataset[:,0],reduced_dataset[:,1])
    plt.title("before training")
    for iteration in range(n_iterations):
        error_hist.append(sess.run(J,feed_dict={X: X_train}))
        training_op.run(feed_dict={X: X_train})
    reduced_dataset = sess.run(hidden, feed_dict={X: X_train})
    plt.figure()
    plt.scatter(reduced_dataset[:,0],reduced_dataset[:,1])
    plt.title("after training")
plt.figure()
plt.title("Reconstruction error")
plt.plot(error_hist)
