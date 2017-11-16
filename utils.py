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

FLAGS = None
COUNT_COLUMN = 6
EVAL_FILE = '/tmp/phuong/ev.csv'
# EVAL_FILE = '/home/tesla/Desktop/LV/tmp/phuonghq/ev.csv'
tf.logging.set_verbosity(tf.logging.INFO)

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
                df = pd.read_csv(filePath, sep=',', header=None, index_col=0)
                list_.append(df)
            frame = pd.concat(list_)
    frame.to_csv(dataset + '.csv', sep=',')
    dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=dataset + '.csv', target_dtype=np.float64, features_dtype=np.float32)
    return dataset


def scale_data(dataset):
    for v in enumerate(dataset):
        v[1][0] = int(v[1][0]) / 16
        v[1][2] = int(v[1][2]) / 20640
        v[1][3] = int(v[1][3]) / 10


def print_file(rootPath, predict):
    predictArr = np.array(predict)

    index = 1

    for subDir in os.listdir(rootPath + '/Data_predict/'):
        sub_dir = rootPath + '/Data_predict/' + subDir
        if (os.path.isdir(sub_dir)):
            for file in sorted(os.listdir(sub_dir)):
                if file is not None:
                    filePath = sub_dir + "/" + file
                    day_of_week = datetime.datetime.strptime(os.path.splitext(file)[0], '%Y-%m-%d').weekday()
                    with open(filePath, 'r') as fin, open('test' + str(day_of_week) + '.csv', 'w') as fout:
                        writer = csv.writer(fout)
                        writer.writerow(["Frame", "Day of week", "Segment", "Velocity", "Velocity", "Predict"])
                        # for row in csv.reader(fin):
                        #   writer.writerow(row[:-1])
                        # remove duplicate column
                        for line in iter(fin.readline, ''):
                            # print(line)
                            fout.write(line.replace('\n', ', ' + str(predictArr[0][index]) + '\n'))
                            index += 1
    today = datetime.datetime.today().weekday()
    # file_plot = 'day' + '2' + '.csv'
    file_plot = 'test' + str(day_of_week) + '.csv'
    return file_plot  # fileplot


def get_segment(file_plot):
    df = pd.read_csv(file_plot, usecols=[2])
    c = df['Segment'].value_counts()
    return c.index[0]


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
    # classes = tf.as_string(tf.argmax(predictions, 1))

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
    tmp_loss = tf.reduce_sum(tf.abs(predictions - tf.reduce_mean(predictions))) / NODE
    loss = tf.reduce_mean(tf.abs(predictions - tf.cast(labels, tf.float32))) / tmp_loss
    mean, op = tf.metrics.mean(loss)
    return mean, op


def plot_segment(tempVeloToMap, frToMap, segment):
    # x = np.array([datetime.datetime(2017, 11, 17, 0, i) for i in range(60)])
    # y = np.random.randint(100, size=x.shape)
    # x = np.array([i for i in range(95)])
    # x = np.array([i for i in frToMap])
    # y = np.array([i for i in tempVeloToMap])
    #
    # # plt.plot(x, y)
    # plt.scatter(x,y)
    # plt.show()
    # plt.figure().savefig('f1.jpg')
    print('1111')
    fig = plt.figure()
    fig.suptitle('Velocity in day of segment', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Segment %s' % segment)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity')
    #
    # ax.text(3, 8, 'boxed italics text in data coords', style='italic',
    #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    #
    # ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
    #
    # ax.text(3, 2, u'unicode:PHUONGHQ')
    #
    # ax.text(0.95, 0.01, 'colored text in axes coords',
    #         verticalalignment='bottom', horizontalalignment='right',
    #         transform=ax.transAxes,
    #         color='green', fontsize=15)
    #

    # ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    # ax.axis([1, 95, 0, 60])
    x = np.array([i for i in frToMap])
    y = np.array([i for i in tempVeloToMap])

    ax.plot(x, y, 'o')

    plt.show()

    fig.savefig('f2.png')
