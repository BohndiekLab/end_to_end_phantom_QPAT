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


for i in range(len(signal)):
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

    # save all inclusion aggregated
    for idx in np.unique(seg):
        if idx > 1:
            target_seg = (seg == idx)
            distances = distance_transform_edt(target_seg)
            inc_fluence_corrected.append(np.mean(bg_sig[target_seg & (distances < DISTANCE_THRESHOLD)]))
            inc_absorptions.append(np.median(mua[target_seg]))
            est_exp_inc_absorptions.append(np.mean(est_mua_exp_img[target_seg & (distances < DISTANCE_THRESHOLD)]))
            est_sim_inc_absorptions.append(np.mean(est_mua_sim_img[target_seg & (distances < DISTANCE_THRESHOLD)]))

bg_absorptions = np.asarray(bg_absorptions)
bg_fluence_corrected = np.asarray(bg_fluence_corrected)
inc_fluence_corrected = np.asarray(inc_fluence_corrected)
inc_absorptions = np.asarray(inc_absorptions)
est_exp_bg_absorptions = np.asarray(est_exp_bg_absorptions)
est_exp_inc_absorptions = np.asarray(est_exp_inc_absorptions)

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 3, wspace=1.5, hspace=2)
gs.update(top=1, bottom=0, left=0, right=1)

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

f1 = fig.add_subfigure(gs[0, 0])
f2 = fig.add_subfigure(gs[1, 0])
f3 = fig.add_subfigure(gs[0, 1])
f4 = fig.add_subfigure(gs[1, 1])
f5 = fig.add_subfigure(gs[0, 2])
f6 = fig.add_subfigure(gs[1, 2])

subfig_regression_line(f1, bg_absorptions, (bg_fluence_corrected - FLUENCE_CALIBRATION_INTERCEPT_BG) / FLUENCE_CALIBRATION_SLOPE_BG,
                       ylabel=r"Background estimate", title=r"GT-$\phi$", color=COLOURS[0],
                       right=0.98, top=0.95, bottom=0.17, num="A")
subfig_regression_line(f2, inc_absorptions, (inc_fluence_corrected - FLUENCE_CALIBRATION_INTERCEPT_BG) / FLUENCE_CALIBRATION_SLOPE_BG,
                       ylabel=r"Inclusion estimate", title=r"GT-$\phi$", color=COLOURS[0],
                       right=0.98, top=0.9, bottom=0.12, num="D",
                       inclusions=True, first=False)

subfig_regression_line(f3, bg_absorptions, est_sim_bg_absorptions,
                       title=r"DL-Sim", color=COLOURS[2], right=0.98, top=0.95, bottom=0.17, num="B",
                       y_axis=False, first=False)
subfig_regression_line(f4, inc_absorptions, est_sim_inc_absorptions,
                       title=r"DL-Sim", color=COLOURS[2], right=0.98, top=0.9, bottom=0.12, num="E",
                       y_axis=False, inclusions=True, first=False)

subfig_regression_line(f5, bg_absorptions, est_exp_bg_absorptions, title=r"DL-Exp",
                       color=COLOURS[1], right=0.9, top=0.95, bottom=0.17, num="C",
                       y_axis=False, first=False)
subfig_regression_line(f6, inc_absorptions, est_exp_inc_absorptions, title="DL-Exp",
                       color=COLOURS[1], right=0.9, top=0.9, bottom=0.12, num="F",
                       y_axis=False, inclusions=True, first=False)

rel_error_gtphi = np.abs((inc_fluence_corrected - FLUENCE_CALIBRATION_INTERCEPT_BG) / FLUENCE_CALIBRATION_SLOPE_BG - inc_absorptions) / inc_absorptions
rel_error_sim = np.abs(est_sim_inc_absorptions - inc_absorptions) / inc_absorptions
rel_error_exp = np.abs(est_exp_inc_absorptions - inc_absorptions) / inc_absorptions

print(mannwhitneyu(rel_error_gtphi, rel_error_sim))
print(mannwhitneyu(rel_error_gtphi, rel_error_exp))
print(mannwhitneyu(rel_error_exp, rel_error_sim))

plt.savefig(fr"suppl_figure3.png", bbox_inches='tight', dpi=300)
plt.close()
