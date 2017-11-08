# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:51:13 2017

@author: lovel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt

import argparse
import sys
import tempfile
import pandas as pd
from six.moves import urllib
import csv
import numpy as np
import tensorflow as tf
import os
from utils import model_fn,load_data,scale_data,prepare_print_file,plot_segment
import datetime

FLAGS = None
tmpArr = []

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
# 0.0001 -0.0002
LEARNING_RATE = 0.000005
file_train = r"C:\Users\lovel\LV\Data_train/2017-05/2017-05-17.csv"
file_test = r"C:\Users\lovel\LV\Data_test/2017-05/2017-05-03.csv"
file_predict = r"C:\Users\lovel\LV\Data_predict/2017-05/2017-05-31.csv"

#file_train = r"/home/tesla/Desktop/LV/Data_train/2017-05/2017-05-17.csv"
#file_test = r"/home/tesla/Desktop/LV/Data_test/2017-05/2017-05-25.csv"
#file_predict = r"/home/tesla/Desktop/LV/2017-05-31.csv"


def main(unused_argv):
    # Load datasets
    rootPath = os.path.abspath(os.path.dirname(__file__))
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=file_train, target_dtype=np.float64, features_dtype=np.float32)
    training_set = load_data(rootPath,'/Data_train/')

    # Test examples
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=file_test, target_dtype=np.float64, features_dtype=np.float32)
    test_set = load_data(rootPath,'/Data_test/')

    # Set of 7 days lastest for which to predict velocity
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=file_predict, target_dtype=np.float64, features_dtype=np.float32)
    prediction_set = load_data(rootPath,'/Data_predict/')

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
    # nn = tf.estimator.Estimator(model_fn=simple_rnn, params=model_params)

    print(test_set[0])
    scale_data(training_set)
    scale_data(test_set)
    scale_data(prediction_set)


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
        # print("Prediction %s: %s" % (i + 1, p['velocity']))
        print("Prediction %s: %s" % (i + 1, p))

        tmpArr.append(p['velocity'])

    print(ev)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])

    # print(training_set)
    # print(test_set)
    # print(prediction_set)

    # print(tmpArr)
    print(training_set[0].shape)
    print(test_set[0].shape)
    print(prediction_set[0].shape)

    ###print predict value to csv file
    plot_segment()
    cur_day_of_week = datetime.datetime.today().weekday()
    print(cur_day_of_week)
    file_print = prepare_print_file(rootPath,'/Data_predict/',cur_day_of_week)
    with open(file_print, 'r') as fin, open('new_predict.csv', 'w') as fout:
    # with open(file_predict, 'r') as fin, open('new_predict.csv', 'w') as fout:

        index = 0
        # fout.write(line.replace('\n', ', ' + 'Predict' + '\n'))
        for line in iter(fin.readline, ''):
            if index == 0:
                fout.write(line.replace('\n', ', ' + str(tmpArr[index]) +',' + 'MSE: ' + str(ev['loss']) +',' + 'RMSE: ' + str(ev['rmse']) +',' + 'MAPE: ' + str(ev['mape'])+',' + 'MASE: ' + str(ev['mase'])  + '\n'))
            else:
                fout.write(line.replace('\n', ', ' + str(tmpArr[index])  + '\n'))
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
