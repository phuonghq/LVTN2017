# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:55:57 2017

@author: lovel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import rnn
import datetime
import pandas as pd

FLAGS = None
tmpArr = []

tf.logging.set_verbosity(tf.logging.INFO)

LSTM_SIZE = 1  # number of hidden layers in each of the LSTM cells
NODE = 100

# create the inference model
# def simple_rnn(features, labels, mode,params):
#     # 0. Reformat input shape to become a sequence
#     x = tf.split(features["x"], 3, 1)
#     print('x={}'.format(x))
#
#     # 1. configure the RNN
#     lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
#     outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float64)
#
#     # slice to keep only the last cell of the RNN
#     outputs = outputs[-1]
#     print('last outputs={}'.format(outputs))
#
#     # output is result of linear activation of last layer of RNN
#     weight = tf.Variable(tf.random_normal([LSTM_SIZE, 1]))
#     bias = tf.Variable(tf.random_normal([1]))
#     predictions = tf.nn.tanh(tf.matmul(outputs, weight)+ bias)
#     labels = tf.nn.tanh(tf.matmul(labels, weight) + bias)
#
#     # 2. Define the loss function for training/evaluation
#     print('labels={}'.format(labels))
#     print('preds={}'.format(predictions))
#     loss = tf.losses.mean_squared_error(labels, predictions)
#     eval_metric_ops = {
#         "rmse": tf.metrics.root_mean_squared_error(labels, predictions)
#     }
#
#     # 3. Define the training operation/optimizer
#     train_op = tf.contrib.layers.optimize_loss(
#         loss=loss,
#         global_step=tf.contrib.framework.get_global_step(),
#         learning_rate=params["learning_rate"],
#         optimizer="SGD")
#
#     # 4. Create predictions
#     predictions_dict = {"predicted": predictions}
#
#     # 5. return ModelFnOps
#     return tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions=predictions_dict,
#         loss=loss,
#         train_op=train_op,
#         eval_metric_ops=eval_metric_ops)
def load_data(rootPath,dir,dataset):
    frame = pd.DataFrame()
    for subDir in os.listdir(rootPath + dir):
        sub_dir = rootPath + dir + subDir
        print(sub_dir)
        if (os.path.isdir(sub_dir)):
            list_ = []
            for filePath in os.listdir(sub_dir):
                filePath = sub_dir + "/" + filePath
                # print(filePath)
                # df = pd.read_csv(filePath,index_col = 0)
                df = pd.read_csv(filePath,sep=',',header=None,index_col=0)

                # print(df)
                list_.append(df)
                # print(list_)
            print(list_)
            frame = pd.concat(list_)
    # frame.to_csv(dataset+ '.csv',sep =',')
    print(frame)
    frame.to_csv(dataset+ '.csv',sep =',')
    dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=dataset + '.csv', target_dtype=np.float64, features_dtype=np.float32)
    return dataset

def scale_data(dataset):
    print(dataset)
    for v in enumerate(dataset):
        v[1][0] = int(v[1][0]) / 16
        v[1][2] = int(v[1][2]) / 20640
        v[1][3] = int(v[1][3]) / 10

    print('test:')
    print(dataset)

def file_to_print(rootPath,dir,cur_day_of_weeks):
    for subDir in os.listdir(rootPath + dir):
        sub_dir = rootPath + dir + subDir
        if (os.path.isdir(sub_dir)):
            for filePath in os.listdir(sub_dir):
                print(filePath)
                day_of_weeks = datetime.datetime.strptime(os.path.splitext(filePath)[0], '%Y-%m-%d').weekday()
                print(day_of_weeks)
                if(day_of_weeks == cur_day_of_weeks):
                    filePath = sub_dir + "/" + filePath
                    print(filePath)
                    return filePath

def model_fn(features, labels, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation
    first_hidden_layer = tf.layers.dense(features["x"], NODE, activation=tf.nn.relu)

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.layers.dense(
        first_hidden_layer, NODE, activation=tf.nn.relu)

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.layers.dense(second_hidden_layer, 1)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"velocity": predictions,})


    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)
    print(labels)
    print(predictions)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32), predictions),
        "mape": mean_absolute_percentage_error(labels,predictions),
        "mase": mean_absolute_scaled_error(labels,predictions)
    }
    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )

def mean_absolute_percentage_error(labels, predictions):
   loss = tf.reduce_mean(tf.abs(tf.cast(labels, tf.float32)-predictions)/predictions) * 100
   mean, op = tf.metrics.mean(loss)
   return mean, op

def mean_absolute_scaled_error(labels,predictions):
    tmp_loss = tf.reduce_sum(tf.abs(predictions - tf.reduce_mean(predictions))) / NODE
    loss = tf.reduce_mean(tf.abs(predictions - tf.cast(labels,tf.float32)))/tmp_loss
    mean, op = tf.metrics.mean(loss)
    return mean, op

def plot_segment():
    # x = np.array([datetime.datetime(2017, 11, 17, 0, i) for i in range(60)])
    # y = np.random.randint(100, size=x.shape)
    x = np.array([i for i in range(95)])
    y = np.random.randint(100, size=x.shape)

    plt.plot(x, y)
    plt.show()