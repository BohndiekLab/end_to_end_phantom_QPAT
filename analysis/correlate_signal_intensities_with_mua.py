import os
import numpy as np
from utils.visualise import plot_regression_line
import matplotlib.pyplot as plt
from scipy.stats import linregress
from utils.data_loading import PalpaitineDataset

BASE_PATH = r"..\mua_fluence/"

# ########################################################
# FIND CALIBRATION CURVE WITH TRAINING DATA
# ########################################################

train_data = PalpaitineDataset(data_path=f"C:/final_data_fluence/training",
                               augment=False, use_all_data=True, experimental_data=True)

train_signals = []
train_muas = []
for item in train_data:
    signal = np.squeeze(item[0]).numpy()
    mua = np.squeeze(item[2]).numpy()
    seg = np.squeeze(item[5])
    train_signals.append(np.percentile(signal[seg==1], 98))
    train_muas.append(np.median(mua[seg==1]))
train_signals = np.asarray(train_signals)
train_muas = np.asarray(train_muas)

slope, intercept, r_value, p_value, std_err = linregress(train_muas, train_signals)

plt.scatter(train_muas, train_signals, c="black", alpha=0.1, label="measurements")
plt.plot(train_muas, intercept + slope * train_muas, 'b', linestyle="dotted", label=f"correlation (R={r_value:.2f})",
        alpha=0.6)
plt.tick_params(direction='in')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.xlabel("Ground Truth Absorption [cm$^{-1}$]")
plt.ylabel("MSOT signal [a.u.]")
plt.legend(loc="upper left")
plt.savefig(f"{BASE_PATH}/calibration_training.png")
plt.close()

print(slope)
print(intercept)
print(r_value)
