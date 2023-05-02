import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from utils.visualise import subfig_regression_line, subfig_rel_error, subfig_abs_error
from utils.regression import get_mua_regression_line, get_fluence_corrected_regression_line

EXPERIMENTAL = False

if EXPERIMENTAL:
    BASE_PATH = r"..\experimental/"
else:
    BASE_PATH = r"..\simulation/"

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

signal_mua_slope, signal_mua_intercept = get_mua_regression_line(BASE_PATH)
bg_slope, bg_intercept = get_fluence_corrected_regression_line(DISTANCE_THRESHOLD)

# Set to true if mua above 2.4 cm-1 should be ignored for the analysis
LOW_ABSORPTION_ONLY = False

# ############################################################################################
# Perform calibration with training data
# ############################################################################################

# bg_slope = 8801.3456983042
# bg_intercept = 832.4797676291034


path = f"{BASE_PATH}/fold_0/data.npz"
estimated_data = np.load(path)
signal = np.squeeze(estimated_data["gt_inputs"])
gt_mua = np.squeeze(estimated_data["gt_muas"])
gt_segmentation = np.squeeze(estimated_data["gt_segmentations"])
fluences = np.squeeze(estimated_data["gt_fluences"])
est_fluence = np.squeeze(estimated_data["est_fluences"])
est_mua = np.squeeze(estimated_data["est_muas"])

num=1
for i in range(1, 5):
    path = fr"{BASE_PATH}\fold_{str(i)}\data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_fluence += np.squeeze(estimated_data["est_fluences"])
        est_mua += np.squeeze(estimated_data["est_muas"])
        num = num + 1

est_fluence = est_fluence / num
est_mua = est_mua / num

bg_fluence_corrected = []
bg_absorptions = []
inc_fluence_corrected = []
inc_absorptions = []
est_phi_bg_absorptions = []
est_phi_inc_absorptions = []
est_mua_bg_absorptions = []
est_mua_inc_absorptions = []


for i in range(len(signal)):
    # extract the values for this image
    bg_sig = np.abs((np.squeeze(signal[i]))) / fluences[i]
    est_phi_img = np.squeeze(signal[i]) / est_fluence[i]
    est_mua_img = np.squeeze(est_mua[i])
    mua = np.squeeze(gt_mua[i])
    seg = np.squeeze(gt_segmentation[i])

    # save all the background aggregates
    bg_seg = (seg == 1)
    bg_fluence_corrected.append(np.median(bg_sig[bg_seg]))
    bg_absorptions.append(np.median(mua[bg_seg]))
    est_phi_bg_absorptions.append(np.median(est_phi_img[bg_seg]))
    est_mua_bg_absorptions.append(np.median(est_mua_img[bg_seg]))

    # save all inclusion aggregated
    for idx in np.unique(seg):
        if idx > 1:
            target_seg = (seg == idx)
            distances = distance_transform_edt(target_seg)
            inc_fluence_corrected.append(np.mean(bg_sig[target_seg & (distances < DISTANCE_THRESHOLD)]))
            inc_absorptions.append(np.median(mua[target_seg]))
            est_phi_inc_absorptions.append(np.mean(est_phi_img[target_seg & (distances < DISTANCE_THRESHOLD)]))
            est_mua_inc_absorptions.append(np.mean(est_mua_img[target_seg & (distances < DISTANCE_THRESHOLD)]))

bg_absorptions = np.asarray(bg_absorptions)
bg_fluence_corrected = np.asarray(bg_fluence_corrected)
inc_fluence_corrected = np.asarray(inc_fluence_corrected)
inc_absorptions = np.asarray(inc_absorptions)
est_phi_bg_absorptions = np.asarray(est_phi_bg_absorptions)
est_phi_inc_absorptions = np.asarray(est_phi_inc_absorptions)
est_mua_bg_absorptions = np.asarray(est_mua_bg_absorptions)
est_mua_inc_absorptions = np.asarray(est_mua_inc_absorptions)

if LOW_ABSORPTION_ONLY:
    inc_fluence_corrected = inc_fluence_corrected[inc_absorptions < 2.5]
    est_phi_inc_absorptions = est_phi_inc_absorptions[inc_absorptions < 2.5]
    est_mua_inc_absorptions = est_mua_inc_absorptions[inc_absorptions < 2.5]
    inc_absorptions = inc_absorptions[inc_absorptions < 2.5]

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

# f7 = fig.add_subfigure(gs[0, 3])
# f8 = fig.add_subfigure(gs[1, 3])
# f9 = fig.add_subfigure(gs[0, 4])
# f10 = fig.add_subfigure(gs[1, 4])

subfig_regression_line(f1, bg_absorptions, (bg_fluence_corrected - bg_intercept) / bg_slope,
                       ylabel=r"Background estimate", title="GT-$\phi$", color=COLOURS[0],
                       right=0.98, top=0.95, bottom=0.17, num="A")
subfig_regression_line(f2, inc_absorptions, (inc_fluence_corrected - bg_intercept) / bg_slope,
                       ylabel=r"Inclusion estimate", title="GT-$\phi$", color=COLOURS[0],
                       right=0.98, top=0.9, bottom=0.12, num="F",
                       inclusions=True, first=False)

subfig_regression_line(f3, bg_absorptions, (est_phi_bg_absorptions - bg_intercept) / bg_slope,
                       title=r"DL-$\hat{\phi}$", color=COLOURS[2], right=0.98, top=0.95, bottom=0.17, num="B",
                       y_axis=False, first=False)
subfig_regression_line(f4, inc_absorptions, (est_phi_inc_absorptions - bg_intercept) / bg_slope,
                       title=r"DL-$\hat{\phi}$", color=COLOURS[2], right=0.98, top=0.9, bottom=0.12, num="G",
                       y_axis=False, inclusions=True, first=False)

subfig_regression_line(f5, bg_absorptions, est_mua_bg_absorptions, title=r"DL-$\hat{\mu_a}$",
                       color=COLOURS[1], right=0.9, top=0.95, bottom=0.17, num="C",
                       y_axis=False, first=False)
subfig_regression_line(f6, inc_absorptions, est_mua_inc_absorptions, title="DL-$\hat{\mu_a}$",
                       color=COLOURS[1], right=0.9, top=0.9, bottom=0.12, num="H",
                       y_axis=False, inclusions=True, first=False)

# subfig_rel_error(f7, [[bg_absorptions, (bg_fluence_corrected - bg_intercept) / bg_slope],
#                       [bg_absorptions, (est_phi_bg_absorptions - bg_intercept) / bg_slope],
#                       [bg_absorptions, est_mua_bg_absorptions]],
#                  color=[COLOURS[0], COLOURS[2], COLOURS[1]], left=-0.05, right=0.85, top=0.95, bottom=0.17, num="D")
#
# subfig_abs_error(f9, [[bg_absorptions, (bg_fluence_corrected - bg_intercept) / bg_slope],
#                       [bg_absorptions, (est_phi_bg_absorptions - bg_intercept) / bg_slope],
#                       [bg_absorptions, est_mua_bg_absorptions]],
#                  color=[COLOURS[0], COLOURS[2], COLOURS[1]], left=0.05, right=0.95, top=0.95, bottom=0.17, num="E")
#
# subfig_rel_error(f8, [[inc_absorptions, (inc_fluence_corrected - bg_intercept) / bg_slope],
#                       [inc_absorptions, (est_phi_inc_absorptions - bg_intercept) / bg_slope],
#                       [inc_absorptions, est_mua_inc_absorptions]],
#                  color=[COLOURS[0], COLOURS[2], COLOURS[1]], left=-0.05, right=0.85, top=0.9, bottom=0.12, num="I")
#
# subfig_abs_error(f10, [[inc_absorptions, (inc_fluence_corrected - bg_intercept) / bg_slope],
#                       [inc_absorptions, (est_phi_inc_absorptions - bg_intercept) / bg_slope],
#                       [inc_absorptions, est_mua_inc_absorptions]],
#                  color=[COLOURS[0], COLOURS[2], COLOURS[1]], left=0.05, right=0.95, top=0.9, bottom=0.12, num="J")

plt.savefig(fr"{BASE_PATH}\figures\all_results.png", bbox_inches='tight', dpi=300)
plt.savefig(fr"{BASE_PATH}\figures\all_results.svg", bbox_inches='tight')
plt.savefig(fr"{BASE_PATH}\figures\all_results.pdf", bbox_inches='tight')
plt.close()
