from tensorflow.contrib import predictor
from utils import *
import os
import tensorflow as tf
import csv
from train import *

FRAME_TO_PLOT = 70

def main(unused_argv):

    # Set of 7 days lastest for which to predict velocity
    prediction_set = load_data(rootPath, '/Data_predict/', 'prediction_set')
    scale_data(prediction_set.data)
    print(prediction_set)

    predict_fn = predictor.from_saved_model(load_export_model_dir())

    print(predict_fn)

    predictions = predict_fn(
        {"x": prediction_set.data})  # x instead of velocity //a dict

    # Print out predictions
    print(predictions)
    ###print predict value to csv file

    file_plot = print_file(rootPath, list(predictions.values()))
    print(file_plot)

    # plot frame by segment
    plot_by_segment(file_plot)

    # plot frame by time
    plot_by_frame(file_plot,FRAME_TO_PLOT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
