# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:55:57 2017

@author: lovel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import csv
import tensorflow as tf
import numpy as np
import os
import datetime
import pandas as pd
import re

# Find root
rootPath = os.path.abspath(os.path.dirname(__file__))

FLAGS = None
EVAL_FILE = rootPath + 'tmp/phuonghq/ev.csv'
tf.logging.set_verbosity(tf.logging.INFO)

TEMP_DIR = rootPath + '/tmp/phuonghq/'
NODE = 100


def load_data(rootPath, dir, dataset):
    frame = pd.DataFrame()
    for subDir in sorted(os.listdir(rootPath + dir)):
        sub_dir = rootPath + dir + subDir
        print(sub_dir)
        if (os.path.isdir(sub_dir)):
            list_ = []
            for filePath in sorted(os.listdir(sub_dir)):
                filePath = sub_dir + "/" + filePath
                print(filePath)
                df = pd.read_csv(filePath, header=None, skiprows=1, index_col=0, low_memory=False)
                df[4] = df[3]
                list_.append(df)
            frame = pd.concat(list_)
    frame.to_csv(rootPath + '/Temp/' + dataset + '.csv', sep=',')
    dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=rootPath + '/Temp/' + dataset + '.csv', target_dtype=np.float64, features_dtype=np.float32)

    return dataset


def scale_data(dataset):
    for v in enumerate(dataset):
        v[1][0] = int(v[1][0]) / 16
        v[1][2] = int(v[1][2]) / 20640
        v[1][3] = int(v[1][3]) / 10


def load_export_model_dir():
    for subDir in os.listdir(TEMP_DIR):
        if subDir is None:
            raise ValueError('Model is not exist!')
        else:
            sub_dir = TEMP_DIR + subDir
            print(sub_dir)
            if (os.path.isdir(sub_dir)):
                if subDir != 'eval':
                    export_dir = sub_dir
                    return export_dir


def print_file(rootPath, predict):
    predictArr = np.reshape(predict, (-1, 1))  # list(n,1) to array
    for subDir in os.listdir(rootPath + '/Data_predict/'):
        sub_dir = rootPath + '/Data_predict/' + subDir
        if (os.path.isdir(sub_dir)):
            start = 1
            print(predictArr.shape[0])
            for file in sorted(os.listdir(sub_dir)):
                if file is not None:
                    filePath = sub_dir + "/" + file
                    print(filePath)
                    day_of_week = datetime.datetime.strptime(os.path.splitext(file)[0], '%Y-%m-%d').weekday()
                    today = datetime.datetime.today().weekday()
                    if day_of_week == today:
                        file_plot = rootPath + '/Temp/' + 'new_' + file  ###
                    df = pd.read_csv(filePath, index_col=0, low_memory=False)
                    length = df.shape[0]
                    print(length)
                    df['Predict'] = predictArr[start:start + length]
                    deltaArr = abs(df['Predict'] - df['Velocity']) / df['Velocity'] * 100
                    df['Delta'] = deltaArr
                    df.to_csv(rootPath + '/Temp/' + 'new_' + file, sep=',')
                    end = start + length
                    start = end

    # file_plot = 'Temp/' + 'new_' + file ###
    return file_plot


def get_segment(file_plot):
    df = pd.read_csv(file_plot, index_col=0, usecols=[0, 2])
    c = df['Segment Id'].value_counts()
    return c.index[1]


def model_fn(features, labels, mode, params):
    """Model function for Estimator."""
    # tanh ko on dinh bang relu
    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation
    first_hidden_layer = tf.layers.dense(features["x"], NODE, activation=tf.nn.relu)  # tf.nn.relu instead of tf.tanh(x)

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.layers.dense(
        first_hidden_layer, NODE, activation=tf.nn.relu)
    # third_hidden_layer
    third_hidden_layer = tf.layers.dense(
        second_hidden_layer, NODE, activation=tf.nn.relu)

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.layers.dense(third_hidden_layer, 1)  # softmax : output depend [0,1]

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # export_outputs = {"classes": tf.estimator.export.ClassificationOutput(classes=classes)}
        export_outputs = {"classes": tf.estimator.export.PredictOutput({"x": predictions})}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"velocity": predictions},
            export_outputs=export_outputs
        )

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)
    # Show in tensorboard
    # tf.summary.scalar("Loss", loss)
    print(labels)
    print(predictions)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "mae": mean_absolute_error(labels, predictions),
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32), predictions),
        "mape": mean_absolute_percentage_error(labels, predictions),
        "mase": mean_absolute_scaled_error(labels, predictions)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )


def mean_absolute_error(labels, predictions):
    loss = tf.reduce_mean(tf.abs(tf.cast(labels, tf.float32) - predictions))
    mean, op = tf.metrics.mean(loss)
    return mean, op


def mean_absolute_percentage_error(labels, predictions):
    loss = tf.reduce_mean(tf.abs(tf.cast(labels, tf.float32) - predictions) / predictions) * 100
    mean, op = tf.metrics.mean(loss)
    return mean, op


def mean_absolute_scaled_error(labels, predictions):
    n = predictions.shape
    tmp_loss = tf.reduce_sum(tf.abs(np.diff(predictions))) / (n - 1)
    # tmp_loss = tf.reduce_sum(tf.abs(predictions - tf.reduce_mean(predictions))) / (n-1)

    loss = tf.reduce_mean(tf.abs(predictions - tf.cast(labels, tf.float32))) / tmp_loss
    mean, op = tf.metrics.mean(loss)
    return mean, op


def plot(tempVeloToMap, varToMap, veloToMap, var, var_flag):
    fig = plt.figure()
    fig.suptitle('Velocity', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    if var_flag == 1:
        ax.set_title('Frame: ' + str(var))
        ax.set_xlabel('Segtment')
    else:
        ax.set_title('Segment: ' + str(var))
        ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity')

    x = np.array([i for i in varToMap])
    y = np.array([i for i in tempVeloToMap])
    z = np.array([i for i in veloToMap])

    plt.scatter(x, y)
    plt.scatter(x, z)
    plt.legend(['Predict', 'Velocity'], loc='upper right')

    plt.show()

    # fig.savefig('f2.png')


def plot_by_segment(file_plot):
    tempVeloToMap = []
    frToMap = []
    veloToMap = []
    seg_plot = get_segment(file_plot)
    print(seg_plot)
    # funtion to get segment

    # append velo
    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:  # for each row in the reader object
            if row['Segment Id'] == str(seg_plot):
                tempVeloToMap.append(row['Predict'])
                frToMap.append(row['Frame'])
                veloToMap.append(row['Velocity'])
    # plot segment
    print(tempVeloToMap)
    print(frToMap)
    print(veloToMap)
    plot(tempVeloToMap, frToMap, veloToMap, seg_plot, 0)


def plot_by_frame(file_plot, frame):
    tempVeloToMap = []
    segToMap = []
    veloToMap = []

    print(frame)
    # funtion to get segment

    # append velo
    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:  # for each row in the reader object
            if row['Frame'] == str(frame) and float(row['Delta']) > 20.0 and float(row['Velocity']) < 10.0:
                tempVeloToMap.append(row['Predict'])
                segToMap.append(row['Segment Id'])
                veloToMap.append(row['Velocity'])
    # plot segment
    print(tempVeloToMap)
    print(segToMap)
    print(veloToMap)
    plot(tempVeloToMap, segToMap, veloToMap, frame, 1)
