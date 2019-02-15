# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:26:21 2019

@author: Lhy
"""

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

#configure the parameters
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

#the progress of the train
def train(mnist):
    x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name ="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    