# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:51:13 2017

@author: phuonghq
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import argparse
import sys
import tempfile
import pandas as pd
from six.moves import urllib
import csv
import numpy as np
import tensorflow as tf
import os
from utils import model_fn,load_data,scale_data

FLAGS = None
DEFAULT_BATCH_SIZE = 100
EXPORT_DIR_BASE = '\\tmp\\phuong\\'
INPUT_COLUMNS = 4
tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
# 0.0001 -0.0002
LEARNING_RATE = 0.000005


def serving_input_receiver_fn():
    """Build the serving inputs."""
    inputs = {"x": tf.placeholder(shape=[None, 4], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def main(unused_argv):
    #remove all file old model
    if os.path.exists(EXPORT_DIR_BASE) == True:
        shutil.rmtree(EXPORT_DIR_BASE)

    # Load datasets
    rootPath = os.path.abspath(os.path.dirname(__file__))

    # Training examples
    training_set = load_data(rootPath,'/Data_train/','training_set')

    # Test examples
    test_set = load_data(rootPath,'/Data_test/','test_set')

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=EXPORT_DIR_BASE)
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, config=run_config)

# scale data
    scale_data(training_set.data)
    scale_data(test_set.data)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_set.data},
        y=np.array(training_set.target),
        num_epochs=None,  # 0
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True)

    # Train
    nn.train(input_fn=train_input_fn, steps=5000)
    # export
    nn.export_savedmodel(EXPORT_DIR_BASE,
        serving_input_receiver_fn=serving_input_receiver_fn)

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_set.data},
        y=np.array(test_set.target),
        num_epochs=1,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])
    f = open(EXPORT_DIR_BASE + 'ev.csv', 'w')
    f.write('MSE: ' + str(ev['loss']) + ',' + 'RMSE: ' + str(
                                ev['rmse']) + ',' + 'MAPE: ' + str(ev['mape']) + ',' + 'MASE: ' + str(ev['mase']) )
    f.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

