# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:51:13 2017

@author: lovel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import random
import argparse
import sys
import tempfile
import pandas as pd
from six.moves import urllib
import csv
import numpy as np
import tensorflow as tf
import os
from utils import model_fn,load_data,scale_data,file_to_print,plot_segment,print_file
import datetime

FLAGS = None
COUNT_COLUMN = 6 #column of data input
tmpArr = []

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
# 0.0001 -0.0002
LEARNING_RATE = 0.000005
file_train = r"C:\Users\lovel\LV\LVTN2017/Data_train/2017-05/2017-05-17.csv"
file_test = r"C:\Users\lovel\LV\LVTN2017/Data_test/2017-05/2017-05-03.csv"
file_predict = r"C:\Users\lovel\LV\LVTN2017/Data_predict/2017-05/2017-05-31.csv"

#file_train = r"/home/tesla/Desktop/LV/Data_train/2017-05/2017-05-17.csv"
#file_test = r"/home/tesla/Desktop/LV/Data_test/2017-05/2017-05-25.csv"
#file_predict = r"/home/tesla/Desktop/LV/2017-05-31.csv"


def main(unused_argv):
    # Load datasets
    rootPath = os.path.abspath(os.path.dirname(__file__))
    training_set = load_data(rootPath,'/Data_train/','training_set')

    # Test examples
    test_set = load_data(rootPath,'/Data_test/','test_set')

    # Set of 7 days lastest for which to predict velocity
    prediction_set = load_data(rootPath,'/Data_predict/','prediction_set')

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
    # nn = tf.estimator.Estimator(model_fn=simple_rnn, params=model_params)

    scale_data(training_set.data)
    scale_data(test_set.data)
    scale_data(prediction_set.data)

    print(training_set[0])

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_set.data},
        y=np.array(training_set.target),
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
        x={"x":test_set.data},
        y=np.array(test_set.target),
        num_epochs=1,
        batch_size=100,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)
    # print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])

    # Print out predictions
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": prediction_set.data},
        num_epochs=1,
        shuffle=False)
    predictions = nn.predict(input_fn=predict_input_fn)
    print(predictions)
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i + 1, p['velocity']))
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
    # plot_segment()
    tempSeg = []
    tempVeloToMap = []
    # cur_day_of_week = datetime.datetime.today().weekday()
    # print(cur_day_of_week)
    file_plot = print_file(rootPath,ev,tmpArr)
    print(file_plot)
    # plot plot_segment.
    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        print(reader)
        for row in reader:
            i =0
            for k, v in row.items():         # get segment_id
                if i % COUNT_COLUMN == 2:
                    tempSeg.append(v)
                i+= 1
        #select random segment to plot
        rand_seg = random.choice(list(tempSeg))
        print(rand_seg)
        print('---------')
    with open(file_plot, 'rb') as f:
        reader = csv.DictReader(f)
        # rows = [row for row in reader for k in row.items() if row['Total_Depth'] != '0']
        for row in reader:  # for each row in the reader object
            if row[2] == rand_seg:
                for h, v in row:
                    tempVeloToMap[h].append(v)


    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = 0
            m = 0
            j = 0
            for k, v in row.items():         # get segment_id
                print('---------')
                # print(rand_seg)
                print(v)
                # print(m)
                print('--------')
                # if i % COUNT_COLUMN == 2:
                #     m = v
                # # if m == rand_seg:
                # if m == 37:
                #
                #     j += 1
                #     if j == 4:
                #         tempVeloToMap.append(v)
                # i += 1
        print(tempVeloToMap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
