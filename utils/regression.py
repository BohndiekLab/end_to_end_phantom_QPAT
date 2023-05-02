import numpy as np
from data_path import DATA_PATH
from scipy.stats import linregress
from utils.data_loading import PalpaitineDataset
from scipy.ndimage import distance_transform_edt


def get_fluence_corrected_regression_line(distance_threshold):

    train_data = PalpaitineDataset(data_path=f"{DATA_PATH}/training",
                                   augment=False, use_all_data=True, experimental_data=True)
    bg_fluence_corrected = []
    inc_fluence_corrected = []
    inc_muas = []
    bg_muas = []
    for item in train_data:
        fluence = np.squeeze(item[4]).numpy()
        signal_bg = np.abs((np.squeeze(item[0]).numpy())) / fluence
        mua = np.squeeze(item[2]).numpy()
        seg = np.squeeze(item[5])
        bg_fluence_corrected.append(np.median(signal_bg[seg == 1]))
        bg_muas.append(np.median(mua[seg == 1]))
        for idx in np.unique(seg):
            if idx > 1:
                target_seg = (seg == idx)
                distances = distance_transform_edt(target_seg)
                inc_fluence_corrected.append(np.mean(signal_bg[target_seg & (distances < distance_threshold)]))
                inc_muas.append(np.median(mua[target_seg]))
    bg_fluence_corrected = np.asarray(bg_fluence_corrected)
    inc_fluence_corrected = np.asarray(inc_fluence_corrected)
    inc_muas = np.asarray(inc_muas)
    bg_muas = np.asarray(bg_muas)

    bg_slope, bg_intercept, bg_r_value, _, _ = linregress(bg_muas, bg_fluence_corrected)
    print("Fluence corrected calibration:")
    print("bg_slope", bg_slope)
    print("bg_intercept", bg_intercept)
    print(bg_r_value)

    print("Inclusion error:")
    print(np.median(np.abs(((inc_fluence_corrected - bg_intercept) / bg_slope) - inc_muas)))
    print(np.median(np.abs(((inc_fluence_corrected - bg_intercept) / bg_slope) - inc_muas) / inc_muas * 100), "%")
    return bg_slope, bg_intercept


def get_mua_regression_line():
    # ########################################################
    # FIND CALIBRATION CURVE WITH TRAINING DATA
    # ########################################################

    train_data = PalpaitineDataset(data_path=f"{DATA_PATH}/training",
                                   augment=False, use_all_data=True, experimental_data=True)

    train_signals = []
    train_muas = []
    for item in train_data:
        signal = np.squeeze(item[0]).numpy()
        mua = np.squeeze(item[2]).numpy()
        seg = np.squeeze(item[5])
        train_signals.append(np.percentile(signal[seg == 1], 98))
        train_muas.append(np.median(mua[seg == 1]))
    train_signals = np.asarray(train_signals)
    train_muas = np.asarray(train_muas)

    slope, intercept, r_value, p_value, std_err = linregress(train_muas, train_signals)

    return slope, intercept
