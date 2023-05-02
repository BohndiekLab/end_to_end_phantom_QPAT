import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.ndimage import distance_transform_edt
from utils.visualise import subfig_regression_line

EXPERIMENTAL = False

EXP_PATH = r"..\experimental/"
SIM_PATH = r"..\simulation/"

COLOURS = [
    "#228833ff",  # GREEN
    "#ee6677ff",  # RED
    "#4477aaff",  # BLUE
    "#ccbb44ff"   # YELLOW
]

# This is a training set-optimised property to discard pixels too deep inside the structures
DISTANCE_THRESHOLD = 12

# signal_mua_slope = 1484.95
# signal_mua_intercept = 313.21

CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = 1484.95, 313.21
FLUENCE_CALIBRATION_SLOPE_BG, FLUENCE_CALIBRATION_INTERCEPT_BG = 8801.3456983042, 832.4797676291034

# ############################################################################################
# Perform calibration with training data
# ############################################################################################

# bg_slope = 8801.3456983042
# bg_intercept = 832.4797676291034


path = f"{EXP_PATH}/fold_0/data.npz"
estimated_data = np.load(path)
signal = np.squeeze(estimated_data["gt_inputs"])
gt_mua = np.squeeze(estimated_data["gt_muas"])
gt_segmentation = np.squeeze(estimated_data["gt_segmentations"])
fluences = np.squeeze(estimated_data["gt_fluences"])
est_mua_exp = np.squeeze(estimated_data["est_muas"])
num=1
for i in range(1, 5):
    path = fr"{EXP_PATH}\fold_{str(i)}\data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua_exp += np.squeeze(estimated_data["est_muas"])
        num = num + 1

est_mua_exp = est_mua_exp / num

path = f"{SIM_PATH}/fold_0/data.npz"
estimated_data = np.load(path)
est_mua_sim = np.squeeze(estimated_data["est_muas"])
num=1
for i in range(1, 5):
    path = fr"{SIM_PATH}\fold_{str(i)}\data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua_sim += np.squeeze(estimated_data["est_muas"])
        num = num + 1

est_mua_sim = est_mua_sim / num

bg_fluence_corrected = []
bg_absorptions = []
inc_fluence_corrected = []
inc_absorptions = []
est_sim_bg_absorptions = []
est_sim_inc_absorptions = []
est_exp_bg_absorptions = []
est_exp_inc_absorptions = []
wavelengths_inc = []
wavelengths_bg = []

wavelengths = np.arange(700, 901, 10)
for i in range(len(signal)):
    wl = wavelengths[i%len(wavelengths)]
    # extract the values for this image
    bg_sig = np.abs((np.squeeze(signal[i]))) / fluences[i]
    est_mua_exp_img = np.squeeze(est_mua_exp[i])
    est_mua_sim_img = np.squeeze(est_mua_sim[i])
    mua = np.squeeze(gt_mua[i])
    seg = np.squeeze(gt_segmentation[i])

    # save all the background aggregates
    bg_seg = (seg == 1)
    bg_fluence_corrected.append(np.median(bg_sig[bg_seg]))
    bg_absorptions.append(np.median(mua[bg_seg]))
    est_exp_bg_absorptions.append(np.median(est_mua_exp_img[bg_seg]))
    est_sim_bg_absorptions.append(np.median(est_mua_sim_img[bg_seg]))
    wavelengths_bg.append(wl)

    # save all inclusion aggregated
    for idx in np.unique(seg):
        if idx > 1:
            target_seg = (seg == idx)
            distances = distance_transform_edt(target_seg)
            inc_fluence_corrected.append(np.mean(bg_sig[target_seg & (distances < DISTANCE_THRESHOLD)]))
            inc_absorptions.append(np.median(mua[target_seg]))
            est_exp_inc_absorptions.append(np.mean(est_mua_exp_img[target_seg & (distances < DISTANCE_THRESHOLD)]))
            est_sim_inc_absorptions.append(np.mean(est_mua_sim_img[target_seg & (distances < DISTANCE_THRESHOLD)]))
            wavelengths_inc.append(wl)

bg_absorptions = np.asarray(bg_absorptions)
bg_fluence_corrected = (np.asarray(bg_fluence_corrected) - FLUENCE_CALIBRATION_INTERCEPT_BG) / FLUENCE_CALIBRATION_SLOPE_BG
inc_fluence_corrected = (np.asarray(inc_fluence_corrected) - FLUENCE_CALIBRATION_INTERCEPT_BG) / FLUENCE_CALIBRATION_SLOPE_BG
inc_absorptions = np.asarray(inc_absorptions)
est_exp_bg_absorptions = np.asarray(est_exp_bg_absorptions)
est_exp_inc_absorptions = np.asarray(est_exp_inc_absorptions)
est_sim_bg_absorptions = np.asarray(est_sim_bg_absorptions)
est_sim_inc_absorptions = np.asarray(est_sim_inc_absorptions)
wavelengths_bg = np.asarray(wavelengths_bg)
wavelengths_inc = np.asarray(wavelengths_inc)

fig = plt.figure(figsize=(12, 8))

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

bg_errors_phi = (bg_fluence_corrected - bg_absorptions)
bg_errors_exp = (est_exp_bg_absorptions - bg_absorptions)
bg_errors_sim = (est_sim_bg_absorptions - bg_absorptions)

inc_errors_phi = (inc_fluence_corrected - inc_absorptions)
inc_errors_exp = (est_exp_inc_absorptions - inc_absorptions)
inc_errors_sim = (est_sim_inc_absorptions - inc_absorptions)

plt.subplot(2, 1, 1)

plt.scatter(wavelengths_bg - 3.33, bg_errors_phi, c=COLOURS[0], marker="o", label="BG GT-$\phi$", alpha=0.3)
plt.scatter(wavelengths_bg, bg_errors_sim, c=COLOURS[2], marker="o", label="BG DL-Sim", alpha=0.3)
plt.scatter(wavelengths_bg + 3.33 , bg_errors_exp, c=COLOURS[1], marker="o", label="BG DL-Exp", alpha=0.3)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.hlines(0, 695, 905, colors="black")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Absolute Error [cm$^{-1}$]")
plt.text(-0.05, 1.0, "A", transform=plt.gca().transAxes,
        size=24, weight='bold')
plt.legend()

plt.subplot(2, 1, 2)
plt.scatter(wavelengths_inc - 3.33, inc_errors_phi, c=COLOURS[0], marker="o", label="INC GT-$\phi$", alpha=0.3)
plt.scatter(wavelengths_inc, inc_errors_sim, c=COLOURS[2], marker="o", label="INC DL-Sim", alpha=0.3)
plt.scatter(wavelengths_inc + 3.33, inc_errors_exp, c=COLOURS[1], marker="o", label="INC DL-Exp", alpha=0.3)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.hlines(0, 695, 905, colors="black")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Absolute Error [cm$^{-1}$]")
plt.text(-0.05, 1.0, "B", transform=plt.gca().transAxes,
        size=24, weight='bold')
plt.legend()

plt.tight_layout()
plt.savefig(fr"suppl_figure7.png", bbox_inches='tight', dpi=300)
plt.close()
