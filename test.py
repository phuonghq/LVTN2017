# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:51:13 2017

@author: lovel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from matplotlib import pyplot as plt

import argparse
import sys
import tempfile
import pandas as pd
from six.moves import urllib
import csv
import numpy as np
import tensorflow as tf
import os
from utils import model_fn

FLAGS = None
tmpArr = []

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
# 0.0001 -0.0002
LEARNING_RATE = 0.000005
file_train = r"C:\Users\lovel\LV\Data_train/2017-05/2017-05-17.csv"
file_test = r"C:\Users\lovel\LV\Data_test/2017-05/2017-05-03.csv"
file_predict = r"C:\Users\lovel\LV\2017-05-31.csv"
csvfile = '2017-05-31.csv'

#file_train = r"/home/tesla/Desktop/LV/Data_train/2017-05/2017-05-17.csv"
#file_test = r"/home/tesla/Desktop/LV/Data_test/2017-05/2017-05-25.csv"
#file_predict = r"/home/tesla/Desktop/LV/2017-05-31.csv"


def main(unused_argv):
    # Load datasets
    rootPath = os.path.abspath(os.path.dirname(__file__))
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=file_train, target_dtype=np.float64, features_dtype=np.float32)
    for subDir in os.listdir(rootPath + "/Data_train"):
        sub_dir = rootPath + "/Data_train/" + subDir
        if (os.path.isdir(sub_dir)):
            for filePath in os.listdir(sub_dir):
                filePath = sub_dir + "/" + filePath
                print(filePath)
                tmp = tf.contrib.learn.datasets.base.load_csv_without_header(
                    filename=filePath, target_dtype=np.float64, features_dtype=np.float32)
                print((np.array(training_set[0])).shape)
                print((np.array(tmp[0])).shape)
                # training_set[0][0] = tf.stack([list(training_set[0][0]), list(tmp[0][0])], axis=1)
                # training_set[0][1] = tf.stack([list(training_set[0][1]), list(tmp[0][1])], axis=1)
                # training_set[1] = tf.stack([list(training_set[1]), list(tmp[1])], axis=1)

                training_set[0]= tf.concat([list(training_set[0]), list(tmp[0])], 0)
                training_set[1]= tf.concat([list(training_set[0]), list(tmp[0])], 0)

                # training_set[0] = np.concatenate(training_set[0],tmp[0])
                # training_set[1] = np.concatenate(training_set[1],tmp[1])
                print(training_set[0])
                print(training_set[1])
            #print(training_set[0].shape)

    # Test examples
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=file_test, target_dtype=np.float64, features_dtype=np.float32)
    for subDir in os.listdir(rootPath + "/Data_test"):
        sub_dir = rootPath + "/Data_test/" + subDir
        if (os.path.isdir(sub_dir)):
            for filePath in os.listdir(sub_dir):
                filePath = sub_dir + "/" + filePath
                print(filePath)
                tmp = tf.contrib.learn.datasets.base.load_csv_without_header(
                    filename=filePath, target_dtype=np.float64, features_dtype=np.float32)
                test_set = np.array(test_set).append(tmp)
            print(test_set[0].shape)

    # Set of 7 examples for which to predict velocity
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=file_predict, target_dtype=np.float64, features_dtype=np.float32)

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
    # nn = tf.estimator.Estimator(model_fn=simple_rnn, params=model_params)

    print(training_set[0].shape)
    print(training_set[1].shape)

    print(test_set[0])
    for v in enumerate(training_set[0]):
        v[1][0] = int(v[1][0]) / 16
        v[1][2] = int(v[1][2]) / 20640
        v[1][3] = int(v[1][3]) / 10
    for v in enumerate(test_set[0]):
        v[1][0] = int(v[1][0]) / 16
        v[1][2] = int(v[1][2]) / 20640
        v[1][3] = int(v[1][3]) / 10
    for v in enumerate(prediction_set[0]):
        v[1][0] = int(v[1][0]) / 16
        v[1][2] = int(v[1][2]) / 20640
        v[1][3] = int(v[1][3]) / 10

    print(training_set[0])
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_set[0]},
        y=np.array(training_set[1]),
        num_epochs=None,  # 0
        batch_size=100,
        shuffle=True)
    print(training_set[0])
    print(test_set[0])
    print(prediction_set[0])
    # Train
    nn.train(input_fn=train_input_fn, steps=5000)

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_set[0]},
        y=np.array(test_set[1]),
        num_epochs=1,
        batch_size=100,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)
    # print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])

    # Print out predictions
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": prediction_set[0]},
        num_epochs=1,
        shuffle=False)
    predictions = nn.predict(input_fn=predict_input_fn)
    print(predictions)
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i + 1, p['velocity']))
        tmpArr.append(p['velocity'])
        # plot_predicted, = plt.plot(predictions, label='predicted')
        # plt.legend(handles=[plot_predicted])
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])
    # print("MAPE Error: %s" % ev["mape"])
    print(training_set)
    print(test_set)
    print(prediction_set)

    print(tmpArr)
    ###print predict value to csv file
    with open(file_predict, 'r') as fin, open('new_predict.csv', 'w') as fout:
        index = 0
        # fout.write(line.replace('\n', ', ' + 'Predict' + '\n'))
        for line in iter(fin.readline, ''):
            fout.write(line.replace('\n', ', ' + str(tmpArr[index]) + '\n'))
            index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_data", type=str, default="", help="Path to the training data.")
    parser.add_argument(
        "--test_data", type=str, default="", help="Path to the test data.")
    parser.add_argument(
        "--predict_data",
        type=str,
        default="",
        help="Path to the prediction data.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
