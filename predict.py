from tensorflow.contrib import predictor
from utils import model_fn,load_data,scale_data,file_to_print,plot_segment,print_file,plot_segment
import os
import tensorflow as tf
from tensorflow.contrib import predictor
import csv
from train import *
COUNT_COLUMN = 5 #column of data input
OFFSET_COLUMN = 1
tmpArr = []
tmpVeloArr = []
TEMP_DIR = '\\tmp\\phuong\\'
def load_export_model_dir():
    for subDir in os.listdir(TEMP_DIR):
        sub_dir = TEMP_DIR + subDir
        print(sub_dir)
        if(os.path.isdir(sub_dir)):
            if sub_dir != 'eval':
               export_dir = sub_dir
               return export_dir
        else:
            raise ValueError('Model is not exist!')
def predict():
    rootPath = os.path.abspath(os.path.dirname(__file__))
    # Set of 7 days lastest for which to predict velocity
    prediction_set = load_data(rootPath,'/Data_predict/','prediction_set')
    scale_data(prediction_set.data)
    print(prediction_set)

    predict_fn = predictor.from_saved_model(load_export_model_dir())

    print(predict_fn)

    predictions=predict_fn(
        {"x": prediction_set.data}) #x instead of velocity //a dict

    print(predictions)
# Print out predictions

    ###print predict value to csv file
    tempVeloToMap = []

    file_plot = print_file(rootPath, list(predictions.values()))
    print(file_plot)
    # plot plot_segment.

    # df = pd.read_csv(file_plot, sep=',', header=None, index_col=0, usecols =[1])
    # grouped_measured_power = df.groupby([' 0'])[' 2']
    # result = grouped_measured_power.aggregate({'min': np.min,
    #                                            'max': np.max,
    #                                            })
    # print(result)
    # # list_.append(df)
    # print(df)
    # with open(file_plot, 'r') as f:
    #     reader = csv.DictReader(f)
    #     num_cols = len(next(reader))
    #     print("Cols %s:" % (num_cols))
    #     print(reader)
    #     print('44444444444444')
    #     sorted_row=[]
    #     for row in reader:
    #         sorted_row += OrderedDict(sorted(row.items(),
    #                                             key=lambda item: reader.fieldnames.index(item[0])))
    #         # sorted_row.append(row.items())
    #     # print(sorted_row)
    #     i =0
    #     for  k,v in sorted_row:         # get segment_id
    #         if i % COUNT_COLUMN == 2:
    #             tempSeg.append(v)
    #         i+= 1
    #     #select random segment to plot
    #     print(tempSeg)
    #     rand_seg = random.choice(list(tempSeg))
    #     print(rand_seg)
    rand_seg = 1

# funtion to get segment

    # append velo
    with open(file_plot, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:  # for each row in the reader object
            if list(row.values())[2] == str(rand_seg):
                tempVeloToMap.append(list(row.values())[4])
    print(tempVeloToMap)

    # plot segment
    plot_segment(tempVeloToMap,rand_seg)

if __name__ == "__main__":
    tf.app.run(predict())
