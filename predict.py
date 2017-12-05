from tensorflow.contrib import predictor
from utils import *
import os
import tensorflow as tf
import csv
from train import *

FRAME_TO_PLOT = 70
VO_THI_SAU_1 = list(range(46431, 46451))
VO_THI_SAU_1.extend(range(113547, 113557))
# VO_THI_SAU_2 = list(range(113547, 113556))

DUONG_3THANG2_1 = list(range(122000, 122080))
DUONG_3THANG2_1.extend(range(122149, 122183))
# DUONG_3THANG2_2 = list(range(122149, 122182))
DUONG_3THANG2_3 = list(range(113270, 113292))

TRUONG_CHINH_1 = list(range(101458,101510))
TRUONG_CHINH_2 = list(set(range(121213,121240)) - set([121219]))

CMT8_1 = list(range(117041,117107))
CMT8_2 = list(range(120628,120641))
CMT8_3 = list(range(114117,114122))

XVNT_1 = list(range(121063,121087))
XVNT_2 = list(range(2537,2555))



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
    # plot_by_segment(file_plot)

    # plot frame by time
    # plot_by_frame(file_plot, FRAME_TO_PLOT)

    segVTS1, frVTS1, predictVTS1, realVTS1 = get_velo_street(VO_THI_SAU_1, file_plot)
    # segVTS2, frVTS2, predictVTS2, realVTS2 = get_velo_street(VO_THI_SAU_2, file_plot)
    seg3THANG2_1, fr3THANG2_1, predict3THANG2_1, real3THANG2_1 = get_velo_street(DUONG_3THANG2_1, file_plot)
    # seg3THANG2_2, fr3THANG2_2, predict3THANG2_2, real3THANG2_2 = get_velo_street(DUONG_3THANG2_2, file_plot)
    seg3THANG2_3, fr3THANG2_3, predict3THANG2_3, real3THANG2_3 = get_velo_street(DUONG_3THANG2_3, file_plot)
    # segTRUONGCHINH_1, frTRUONGCHINH_1, predictTRUONGCHINH1, realTRUONGCHINH1= get_velo_street(TRUONG_CHINH_1, file_plot)
    # segTRUONGCHINH_2, frTRUONGCHINH_2, predictTRUONGCHINH2, realTRUONGCHINH2 = get_velo_street(TRUONG_CHINH_2, file_plot)
    # segCMT8_1, frCMT8_1, predictCMT8_1, realCMT8_1 = get_velo_street(CMT8_1, file_plot)
    # segCMT8_2, frCMT8_2, predictCMT8_2, realCMT8_2 = get_velo_street(CMT8_2, file_plot)
    # segCMT8_3, frCMT8_3, predictCMT8_3, realCMT8_3 = get_velo_street(CMT8_3, file_plot)
    # segXVNT_1, frXVNT_1, predictXVNT_1, realXVNT_1 = get_velo_street(XVNT_1, file_plot)
    # segXVNT_2, frXVNT_2, predictXVNT_2, realXVNT_2 = get_velo_street(XVNT_2, file_plot)

    plot_by_street(segVTS1, frVTS1, predictVTS1, realVTS1, 'Vo Thi Sau', 'From Dinh Tien Hoang To Nam Ky Khoi Nghia',46450)
    # plot_by_street(segVTS2, frVTS2, predictVTS2, realVTS2, 'Vo Thi Sau',' From Nam Ky Khoi Nghia To Vong Xoay Dan Chu')
    plot_by_street(seg3THANG2_1, fr3THANG2_1, predict3THANG2_1, real3THANG2_1, '3 THANG 2', 'Vong Xoay Dan Chu To Le Dai Hanh',122079)
    # plot_by_street(seg3THANG2_2, fr3THANG2_2, predict3THANG2_2, real3THANG2_2, '3 THANG 2', 'Le Dai Hanh To Hong Bang')
    plot_by_street(seg3THANG2_3, fr3THANG2_3, predict3THANG2_3, real3THANG2_3, '3 THANG 2', 'Ly Thai To to Vong Xoay Dan Chu','None')
    # plot_by_street(segTRUONGCHINH_1, frTRUONGCHINH_1, predictTRUONGCHINH1, realTRUONGCHINH1, 'TRUONG CHINH Street 1')
    # plot_by_street(segTRUONGCHINH_2, frTRUONGCHINH_2, predictTRUONGCHINH2, realTRUONGCHINH2, 'TRUONG CHINH Street 2')
    # plot_by_street(segCMT8_1, frCMT8_1, predictCMT8_1, realCMT8_1, 'CMT8 Street 1')
    # plot_by_street(segCMT8_2, frCMT8_2, predictCMT8_2, realCMT8_2, 'CMT8 Street 2')
    # plot_by_street(segCMT8_3, frCMT8_3, predictCMT8_3, realCMT8_3, 'CMT8 Street 3')
    # plot_by_street(segXVNT_1, frXVNT_1, predictXVNT_1, realXVNT_1, 'XVNT Street 1')
    # plot_by_street(segXVNT_2, frXVNT_2, predictXVNT_2, realXVNT_2, 'XVNT Street 2')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
