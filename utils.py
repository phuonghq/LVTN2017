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

def street_plot(name_street,file_plot,frameRequest,name,description,isArima):
    velo, count_seg = get_velocity(name_street, file_plot, frameRequest)
    #name_street+'_name', name_street+'_url' = plot_by_street(name_street, velo, name, count_seg, frameRequest, description)
    name,url= plot_by_street(name_street, velo, name, count_seg, frameRequest, description,isArima)
    print(name,url)
    return {'name': str(name) , 'url': str(url) }
   
def load_export_model_dir(frRequest):
    for subDir in os.listdir(TEMP_DIR +str(frRequest)):
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

    df = pd.read_csv(filePath, header=None, index_col=None,
                     usecols=use_cols, low_memory=False)

    df = df[use_cols]

    if flag == 1:
        seg_df = pd.read_csv(filePath, header=None,
                             usecols=[0], low_memory=False)
        segment_id = np.array(seg_df[0])

    list_.append(df)
    frame = pd.concat(list_)

    frame.to_csv(rootPath + '/Temp/' + dataset + '.csv', sep=',', index=False)
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
                    df = pd.read_csv(filePath, header=None, index_col=0, low_memory=False)
                    length = df.shape[0]

                    count_col = len(df.columns)
                    predictArr = predictArr[1:]
                    df[count_col + 1] = predictArr
                    print(df[count_col + 1])
                    
                    # print(frRequest)
                    # print(df[count_col + 1])
                    #  print(df[frRequest])
                    deltaArr = abs(df[count_col + 1] - df[frRequest]) / df[frRequest] * 100
                    df[count_col + 2] = deltaArr
                    print(df[count_col + 2])
                   
                    df.to_csv(rootPath + '/Temp/' + 'test_' + file, sep=',')
    file_plot = rootPath + '/Temp/' + 'test_' + file  ###
    
    return file_plot, df


def get_segment(file_plot):
    df = pd.read_csv(file_plot, index_col=0, usecols=[0, 2])
    c = df['Segment Id'].value_counts()
    return c.index[1]


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


def newPlot(veloToMap, seg_plot):
    fig = plt.figure()
    fig.suptitle('Velocity', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)

    ax.set_title('Segment: ' + str(seg_plot))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity')
    ax.tick_params(axis='y', which='major', pad=15)

    tmp = veloToMap
    x = list(range(0, np.array(len(veloToMap))))
    real = tmp[:-1]
    real = real + [real[-1]]
    # real.extend([tmp[-2]])

    predict = tmp[:-2]
    # print(predict)
    predict = [predict[0]] + predict
    predict.extend([tmp[-1]])

    real = np.array([i for i in real])

    predict = np.array([i for i in predict])
    # plt.xticks(np.arange(min(x), max(x)+1, 5.0))
    plt.yticks(np.arange(min(real), max(real) + 1, 5.0))
    plt.plot(x, predict)
    plt.plot(x, real)
    plt.legend(['Predict', 'Real'], loc='upper left')
    # plt.show()
    # print(fig)
    name = 'linePlot'
    url = rootPath + '/Output/Images/' + 'linePlot.png'
    fig.savefig(url)
    return name, url


def linePlotBySeg(file_plot, frameRequest, seg_plot):
    veloToMap = []

    # funtion to get segment

    # append velo
    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:  # for each row in the reader object
            seg_id = Decimal(row['0'])
            if seg_id == seg_plot:
                row = {int(k): float(v) for k, v in row.items()}
                veloToMap = row
    tmp = veloToMap.values()
    # print(tmp,frameRequest)
    velo = tmp[1:frameRequest + 1]

    velo.append(str(tmp[-2]))
    return newPlot(velo, seg_plot)


def get_velocity(arrSeg, file_plot, frameRequest):
    veloToMap = []
    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:  # for each row in the reader object
            #print(row)
            seg_id = int(float(row['0']))
            if seg_id in arrSeg:
                row = {int(k): float(v) for k, v in row.items()}
                tmp = row
                val = tmp.values()
                temp = val[1: frameRequest + 1]
                temp.append(val[-2])
                veloToMap.append(np.array(temp))

    veloToMap = np.array(veloToMap)
    n_segment = len(veloToMap)
    veloToMap = veloToMap.ravel()
    # print(len(veloToMap))
    # print(veloToMap.shape)
    return veloToMap, n_segment
    # plot_by_street(arrAfterSeg,arrFr,arrVelo,name_street)


def init_plot_by_street(x, val, location, delta, n_segment, frRequest, name,isArima):
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



    zi = np.reshape(val, (-1, frRequest + 1))
    # print(zi)
    zi = np.ma.masked_where(zi <= 0.00, zi)

    cmap = plt.cm.OrRd
    cmap.set_bad(color='white')
    # print(zi)
    plt.imshow(zi, vmin=0, vmax=60, origin='lower',
               extent=[0, n_segment, 0, frRequest], aspect="auto", cmap='jet_r', interpolation='None')
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


def plot_by_street(x, val, location, n_segment, frRequest, name,isArima):
    delta = max(abs(i - j) for (i, j) in zip(x[1:], x[:-1]))
    return init_plot_by_street(x, val, location, delta - 1, n_segment, frRequest, name,isArima)
