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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from decimal import Decimal

# Find root
rootPath = os.path.abspath(os.path.dirname(__file__))

FLAGS = None
EVAL_FILE = rootPath + 'Model/phuonghq/ev.csv'
tf.logging.set_verbosity(tf.logging.INFO)

TEMP_DIR = rootPath + '/Model/phuonghq/'
IMG_DIR = rootPath + '/Output'
# OFFSET = 10
MAX_FRAME = 97


def street_plot(name_street, file_plot, frameRequest, name, description, isArima):
    velo, count_seg = get_velocity(name_street, file_plot, frameRequest)
    # name_street+'_name', name_street+'_url' = plot_by_street(name_street, velo, name, count_seg, frameRequest, description)
    name, url = plot_by_street(name_street, velo, name, count_seg, frameRequest, description, isArima)
    print(name, url)
    return {'name': str(name), 'url': str(url)}


def load_export_model_dir(frRequest):
    for subDir in os.listdir(TEMP_DIR + str(frRequest)):
        if subDir is None:
            raise ValueError('Model is not exist!')
        else:
            sub_dir = TEMP_DIR + '/' + str(frRequest)
            print(sub_dir)
            if (os.path.isdir(sub_dir)):
                export_dir = sub_dir + "/" + os.listdir(sub_dir)[0]
                return export_dir


def load_data(rootPath, dir, dataset, frRequest, offset, flag):
    if offset > frRequest:
        use_cols = list(range(MAX_FRAME - offset + frRequest - 1, MAX_FRAME))
        use_cols.extend(range(1, frRequest + 1))
        print(use_cols)
    else:
        use_cols = list(range(frRequest - offset, frRequest + 1))
        print(use_cols)
    frame = pd.DataFrame()
    segment_id = []
    for subDir in sorted(os.listdir(rootPath + dir)):
        sub_dir = rootPath + dir + subDir
    print(sub_dir)
    if (os.path.isdir(sub_dir)):
        list_ = []
    for filePath in sorted(os.listdir(sub_dir)):
        filePath = sub_dir + "/" + filePath
    print(filePath)

    df = pd.read_csv(filePath, header=None,
                     usecols=use_cols)
    print(df)
    print("__________")
    # df = df[use_cols]

    if flag == 1:
        seg_df = pd.read_csv(filePath, header=None,
                             usecols=[0])
        segment_id = np.array(seg_df[0])
    print('DDDDD')
    print(df)

    columns = df.columns.tolist()
    index_i = columns.index(frRequest)
    columns = columns[(index_i + 1):] + columns[:(index_i + 1)]
    df = df[columns]
    print(df)
    df.to_csv(rootPath + '/Temp/' + dataset + '.csv', sep=',', index=False, header=None)
    dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=rootPath + '/Temp/' + dataset + '.csv', target_dtype=np.float32, features_dtype=np.float32)
    if flag == 1:
        return dataset, segment_id
    else:
        return dataset


def scale_data(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    scaled = scaler.transform(dataset)

    print('scaled={}'.format(scaled))
    return scaled


def print_file(rootPath, predict, frRequest):
    predictArr = np.reshape(predict, (-1, 1))  # list(n,1) to array
    for subDir in os.listdir(rootPath + '/Predict/'):
        sub_dir = rootPath + '/Predict/' + subDir
        if (os.path.isdir(sub_dir)):
            print(predictArr.shape[0])
            for file in sorted(os.listdir(sub_dir)):
                if file is not None:
                    filePath = sub_dir + "/" + file
                    print(filePath)
                    df = pd.read_csv(filePath, header=None)
                    length = df.shape[0]

                    count_col = len(df.columns)
                    df[count_col + 1] = predictArr

                    # print(frRequest)
                    # print(df[count_col + 1])
                    #  print(df[frRequest])
                    # deltaArr = abs(df[count_col + 1] - df[frRequest]) / df[frRequest] * 100
                    # df[count_col + 2] = deltaArr
                    # print(df[count_col + 2])

                    df.to_csv(rootPath + '/Temp/' + 'test_' + file, sep=',', index=False, header=None)

    file_plot = rootPath + '/Temp/' + 'test_' + file  ###

    return file_plot, df


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(100 * np.abs((y_true - y_pred)) / y_true)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_scaled_error(y_true, y_pred):
    # y_true, y_pred = np.array(y_true, y_pred)
    # n = len(y_pred)
    d = np.abs(y_pred - np.mean(y_pred))

    errors = np.abs(y_true - y_pred)
    return np.mean(np.abs(errors / np.mean(np.abs(d))))


def newPlot(predictDF, realDF, ARIMADF, seg_plot):
    fig = plt.figure()
    fig.set_size_inches(12, 7, forward=True)
    fig.suptitle('Velocity', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.9)

    ax.set_title('Segment: ' + str(seg_plot))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity')
    # ax.tick_params(axis='y', which='major', pad=15)

    x = list(range(0, 96))

    # plt.yticks(np.arange(min(realDF.values), max(real) + 1, 5.0))
    plt.plot(x, predictDF.values)
    plt.plot(x, realDF.values)
    plt.plot(x, ARIMADF.values)
    plt.legend(['Predict', 'Real', 'ARIMA'], bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    # plt.show()
    # print(fig)
    name = 'linePlot'
    url = rootPath + '/Output/Images/' + 'linePlot.png'
    fig.savefig(url)
    fig.close()
    return name, url


def linePlotBySeg(file_plot, seg_plot):
    predictVeloToMap = []
    realVeloToMap = []
    ARIMAVeloToMap = []
    real_value_file = rootPath + '/Predict/Predict/tmp_predict.csv'
    ARIMA_file = rootPath + '/Temp/final_data_ARIMA.csv'
    # funtion to get segment
    predict_DF = pd.read_csv(file_plot, header=None,
                             index_col=[0])
    real_DF = pd.read_csv(real_value_file, header=None,
                          index_col=[0])
    ARIMA_DF = pd.read_csv(ARIMA_file, header=None,
                           index_col=[0])
    predictVeloToMap = predict_DF.loc[seg_plot]
    realVeloToMap = real_DF.loc[seg_plot]
    ARIMAVeloToMap = ARIMA_DF.loc[seg_plot]
    return newPlot(predictVeloToMap, realVeloToMap, ARIMAVeloToMap, seg_plot)


def get_velocity(arrSeg, file_plot, frameRequest):
    veloToMap = []
    segment = []
    df = pd.read_csv(file_plot, header=None,
                     index_col=[0])

    for row in df.iterrows():
        seg_id = int(float(row[0]))
        if seg_id in arrSeg:
            segment.append(seg_id)

    for row in df.iterrows():
        val = int(float(row[0]))
        if val in segment:
            tmp = row[1:]
            # tmp.append
            veloToMap.append(tmp)

    veloToMap = np.array(veloToMap)
    n_segment = len(segment)

    veloToMap = veloToMap.ravel()

    return veloToMap, n_segment
    # plot_by_street(arrAfterSeg,arrFr,arrVelo,name_street)


def init_plot_by_street(x, val, location, delta, n_segment, frRequest, name, isArima):
    y = list(range(0, len(x)))
    # print(x)
    # print(val)
    # print('delta: %d' % delta)
    fig = plt.figure()
    # fig.suptitle(location, fontsize=12, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Street: ' + location)
    ax.set_xlabel('Segment')
    ax.set_ylabel('Frame')

    # get count for línpace

    # DELTA = 113547 - 46450
    # x_ = []
    # for value in x:
    #  if index != 'None' and value > index:
    #     value = value - delta
    # x_.append(value)

    # xi, yi = np.linspace(min(x_), max(x_), max(x_) - min(x_) + 1), np.linspace(min(y), max(y), max(y) - min(y) + 1)



    zi = np.reshape(val, (n_segment, int(len(val) / n_segment)))  # must be 2D array
    # print(zi)
    zi = np.ma.masked_where(zi <= 0.00, zi)

    cmap = plt.cm.OrRd
    cmap.set_bad(color='white')
    # print(zi)
    plt.imshow(zi, vmin=0, vmax=60, origin='lower',
               extent=[0, n_segment, 0, 96], aspect="auto", cmap='jet_r', interpolation='None')
    plt.subplots_adjust(hspace=0.5)
    plt.colorbar()
    # plt.show()

    path = rootPath + IMG_DIR + location + '/'
    dir = os.path.dirname(path)
    # print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
        #   print('111')
    if isArima == False:
        url = rootPath + '/Output/Images/' + location + '.png'
    else:
        url = rootPath + '/ARIMA/Images/' + location + '.png'
    fig.savefig(url)
    return name, url


def plot_by_street(x, val, location, n_segment, frRequest, name, isArima):
    delta = max(abs(i - j) for (i, j) in zip(x[1:], x[:-1]))
    return init_plot_by_street(x, val, location, delta - 1, n_segment, frRequest, name, isArima)
