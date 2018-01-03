from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import PredictionType

from math import sqrt
from utils import *
import numpy as np
import tensorflow as tf
import os
import shutil
from tensorflow.python.estimator.export import export

FLAGS = None

# tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
# LEARNING_RATE = 0.015  # 0.015
EXPORT_DIR_BASE = rootPath + '/Model/phuonghq/'
STEP_SIZE = 9000  # 7000
VO_THI_SAU_1 = list(range(46431, 46451))
VO_THI_SAU_1.extend(range(113547, 113557))
# VO_THI_SAU_2 = list(range(113547, 113556))

DUONG_3THANG2_1 = list(range(122000, 122080))
DUONG_3THANG2_1.extend(range(122149, 122183))
# DUONG_3THANG2_2 = list(range(122149, 122182))
DUONG_3THANG2_3 = list(range(113270, 113292))

TRUONG_CHINH_1 = list(range(101458, 101510))
TRUONG_CHINH_2 = list(set(range(121213, 121240)) - set([121219]))

CMT8_1 = list(range(117041, 117107))
CMT8_2 = list(range(120628, 120641))
CMT8_3 = list(range(114117, 114122))

XVNT_1 = list(range(121063, 121087))
XVNT_2 = list(range(2537, 2555))

DBP = list(range(88084, 88164))


def serving_input_receiver_fn():
    """Build the serving inputs."""

    # feature_spec = {'x':tf.FixedLenFeature(shape=[None,10],dtype= tf.int64) }
    feature_spec = {'x': tf.placeholder(shape=[None,10],dtype=tf.string)}
    return tf.contrib.learn.utils.input_fn_utils.build_default_serving_input_fn(feature_spec)()


def main(frameRequest, offset, learning_rate):
    export_model = EXPORT_DIR_BASE + '/' + str(frameRequest)
    # remove all file old model
    if os.path.exists(EXPORT_DIR_BASE):
       shutil.rmtree(EXPORT_DIR_BASE, ignore_errors=True)
    # Load datasets
    rootPath = os.path.abspath(os.path.dirname(__file__))

    # frameRequest = 70
    frameRequest += 1  # data is counting from 1-96

    training_set = load_data(rootPath, '/Train/', 'training_set', frameRequest, offset, 0)
    print("train: ", training_set)
    # Test examples
    test_set = load_data(rootPath, '/Test/', 'test_set', frameRequest, offset, 0)
    print("test: ",test_set)
    scale_data(training_set.data)
    scale_data(test_set.data)

    # Set model params
    # model_params = {"learning_rate": LEARNING_RATE}
    xc = tf.contrib.layers.real_valued_column("x")

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        save_summary_steps=500,
        model_dir=EXPORT_DIR_BASE,
        save_checkpoints_steps=1000,
        save_checkpoints_secs=None,
        session_config=None,
        keep_checkpoint_max=25,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=200
    )

    nn = tf.contrib.learn.DynamicRnnEstimator(problem_type=constants.ProblemType.LINEAR_REGRESSION,
                                              prediction_type=PredictionType.SINGLE_VALUE,
                                              sequence_feature_columns=[xc],
                                              context_feature_columns=None,
                                              num_units=10,
                                              cell_type='lstm',
                                              optimizer='SGD',
                                              learning_rate=learning_rate,
                                              # dropout_keep_probabilities = [0.5,0.5],
                                              config=run_config
                                              )  # 0.05
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        eval_steps=1
        # every_n_steps=10,
        # early_stopping_metric="loss",
        # early_stopping_metric_minimize=True,
        # early_stopping_rounds=10
    )
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        batch_size=50,
        shuffle=True)
    print("train input fn: ", train_input_fn)
    # Train
    nn.fit(input_fn=train_input_fn, steps=STEP_SIZE, monitors=[validation_monitor])

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        batch_size=50,
        num_epochs=1,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)
    # saved_model_dir = nn.export(export_model,
    #
    #                             input_feature_key=None,
    #                             use_deprecated_input_fn=True,
    #                             signature_fn=None,
    #                             prediction_key=None,
    #                             default_batch_size=100,
    #                             exports_to_keep=100,
    #                             checkpoint_path=None
    #                             # input_fn=my_input_fn,
    #                             )
    # signature_fn=my_signature_fn

   # nn.export_savedmodel(export_model, serving_input_receiver_fn)

    # # print('ev %s' % ev)
 # print("Root Mean Squared Error: %s" % ev["rmse"])
    prediction_set, segment_id = load_data(rootPath, '/Predict/', 'prediction_set', int(frameRequest), int(offset), 1)

    # print(prediction_set.data[:5])
    scale_data(prediction_set.data)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(prediction_set.data)},
        num_epochs=1,
        shuffle=False)

    predictions = nn.predict(input_fn=predict_input_fn)

    output = []
    for i, p in enumerate(predictions):
        output.append(p["scores"])
        if i < 2:
            print("Prediction %s: %s" % (i + 1, p))

    print(prediction_set.target[1:5])
    print(output[1:5])
    delta_value = abs(output - prediction_set.target) / prediction_set.target * 100
    # print(delta_value[1:])

    segDF = pd.DataFrame(segment_id)
    real_veloDF = pd.DataFrame(prediction_set.target[1:])
    predict_veloDF = pd.DataFrame(output[1:])
    deltaDF = pd.DataFrame(delta_value[1:])
    rmse = sqrt(mean_squared_error(np.array(prediction_set.target), output))
    mae = mean_absolute_error(np.array(prediction_set.target), output)

    mape = mean_absolute_percentage_error(prediction_set.target, output)
    mase = mean_absolute_scaled_error(prediction_set.target, output)

    print("Root Mean Squared Error: %s" % rmse)
    print("Mean Absolutely Error: %s" % mae)
    print("Mean Absolutely Percentage Error: %s" % mape)
    print("Mean Absolutely Scaled Error: %s" % mase)
    # f = open(EXPORT_DIR_BASE + 'ev.csv', 'w')
    # f.write('MAE: ' + str(mae) + ',' + 'RMSE: ' + str(
    #     rmse) + ',' + 'MAPE: ' + str(mape) + ',' + 'MASE: ' + str(mase))
    # f.close()
    file_plot, df_out = print_file(rootPath, output, frameRequest)
    print(file_plot)
    # dictLinePlot = {'name': linePlotName,'url':linePlotUrl}
   
    # velo, count_seg = get_velocity(VO_THI_SAU_1, file_plot, frameRequest)
    
    
    
    
    dictCMT8 = street_plot(CMT8_1,file_plot,frameRequest,'CMT8','Cach Mang Thang 8',False)
    dictVTS = street_plot(VO_THI_SAU_1,file_plot,frameRequest,'VOTHISAU','Vo Thi Sau',False)
    dict3THANG2 = street_plot(DUONG_3THANG2_1,file_plot,frameRequest,'3_THANG_2','3 Thang 2',False)
    dictDBP = street_plot(DBP,file_plot,frameRequest,'DBP','Dien Bien Phu',False)
    dictPlot = {'CMT8': dictCMT8, 'VTS': dictVTS, '3_THANG_2': dict3THANG2, 'DBP': dictDBP}
    
    # print(dictPlot)
    #ARIMA
    ARIMA_dir = rootPath + '/ARIMA/Result/'
    ARIMAError_dir = rootPath + '/ARIMA/Evaluate/'
    arima_plot = ARIMA_dir + 'Frame_' + str(frameRequest) + '.csv'
    arima_CMT8 = street_plot(CMT8_1,arima_plot,frameRequest,'CMT8','Cach Mang Thang 8',True)
    arima_VTS = street_plot(VO_THI_SAU_1,arima_plot,frameRequest,'VOTHISAU','Vo Thi Sau',True)
    arima_3THANG2 = street_plot(DUONG_3THANG2_1,arima_plot,frameRequest,'3_THANG_2','3 Thang 2',True)
    arima_DBP = street_plot(DBP,arima_plot,frameRequest,'DBP','Dien Bien Phu',True)
    arima_Dict_Plot = {'CMT8': arima_CMT8, 'VTS': arima_VTS, '3_THANG_2': arima_3THANG2, 'DBP': arima_DBP}
    data_frame = pd.concat([segDF, real_veloDF, predict_veloDF, deltaDF], axis=1)
    print(data_frame)
   
    ARIMA_df = pd.read_csv(ARIMA_dir + 'Frame_' + str(frameRequest) +  '.csv', usecols =[0,frameRequest,97,98],header=None, low_memory=False)
    print(dictPlot)
    ARIMA_error = pd.read_csv(ARIMAError_dir + 'Frame_' + str(frameRequest)+'_ev' + '.csv',header=None,skiprows=[1])
    ARIMA_error = ARIMA_error.values
    print(ARIMA_error)
    print(dictPlot)
    print(arima_Dict_Plot)
    print(file_plot)
   
    #if frameRequest == 95:
    return data_frame, ARIMA_df,dictPlot,arima_Dict_Plot,file_plot, rmse, mae, mape, mase, ARIMA_error[0][1],ARIMA_error[0][0],ARIMA_error[0][2],ARIMA_error[0][3]
    #else:
     #   return main(frameRequest + 1, offset, learning_rate)


if __name__ == "__main__":
    with tf.device('/gpu:0'):
        parser = argparse.ArgumentParser()
        parser.register("type", "bool", lambda v: v.lower() == "true")
        FLAGS, unparsed = parser.parse_known_args()
        parser.add_argument('--frame', metavar='path', required=True,
                            help='the path to frame')
        parser.add_argument('--history', metavar='path', required=True,
                            help='path to history')
        parser.add_argument('--learning_rate', metavar='path', required=True,
                            help='path to learning_rate')
        args = parser.parse_args()
        main(frameRequest=int(args.frame), offset=int(args.history), learning_rate=float(args.learning_rate))
        # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)