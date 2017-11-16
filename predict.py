from tensorflow.contrib import predictor
from utils import *
import os
import tensorflow as tf
from tensorflow.contrib import predictor
import csv
from train import *

COUNT_COLUMN = 5  # column of data input
OFFSET_COLUMN = 1
tmpArr = []
tmpVeloArr = []
TEMP_DIR = '\\tmp\\phuong\\'
# TEMP_DIR = '/home/tesla/Desktop/LV/tmp/phuonghq/'


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


def main(unused_argv):
    rootPath = os.path.abspath(os.path.dirname(__file__))
    # Set of 7 days lastest for which to predict velocity
    prediction_set = load_data(rootPath, '/Data_predict/', 'prediction_set')
    scale_data(prediction_set.data)
    print(prediction_set)

    predict_fn = predictor.from_saved_model(load_export_model_dir())

    print(predict_fn)

    predictions = predict_fn(
        {"x": prediction_set.data})  # x instead of velocity //a dict

    print(predictions)
    # Print out predictions

    ###print predict value to csv file
    tempVeloToMap = []
    frToMap = []
    veloToMap = []
    file_plot = print_file(rootPath, list(predictions.values()))
    print(file_plot)

    seg_plot = get_segment(file_plot)
    print(seg_plot)
    # funtion to get segment

    # append velo
    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:  # for each row in the reader object
            if row['Segment'] == str(seg_plot):
                tempVeloToMap.append(row['Predict'])
                frToMap.append(row['Frame'])
                veloToMap.append(row['Velocity'])
    # plot segment
    print(tempVeloToMap)
    print(frToMap)
    print(veloToMap)
    plot_segment(tempVeloToMap, frToMap, seg_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
