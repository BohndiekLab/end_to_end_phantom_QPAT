# #######################################################################################
# This script
# #######################################################################################

import os
import string
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = get_mua_regression_line(BASE_PATH)
FLUENCE_CALIBRATION_SLOPE_BG, FLUENCE_CALIBRATION_INTERCEPT_BG = get_fluence_corrected_regression_line(DISTANCE_THRESHOLD)

SPACING = 0.10666666667
points = [
    [0, 287, 150, 150],  # 10
    [0, 287, 126, 126],  # 31
    [27, 250, 287, 0],   # 52
    [0, 287, 287, 0],    # 73
    [0, 287, 118, 118],  # 94
    [0, 287, 210, 0],    # 115
    [0, 275, 133, 286],  # 136
    [85, 85, 0, 287],  # 157
    [0, 287, 50, 50],  # 178
    [0, 287, 143, 143],  # 199
    [100, 30, 0, 287],  # 220
    [42, 42, 0, 287],    # 241
    [0, 170, 170, 0],    # 262
    [250, 250, 0, 287],    # 283
    [0, 287, 196, 196],  # 304
    [100, 100, 0, 287],  # 325
    [0, 287, 198, 198],  # 346
    [119, 119, 0, 287]   # 367
]

IDXS = np.arange(0, 18, 1)

print("Loading data...")
path = f"{BASE_PATH}/fold_0/data.npz"
estimated_data = np.load(path)
signal = np.abs(np.squeeze(estimated_data["gt_inputs"]))
gt_mua = np.squeeze(estimated_data["gt_muas"])
gt_segmentation = np.squeeze(estimated_data["gt_segmentations"])
est_mua = np.squeeze(estimated_data["est_muas"])
gt_fluence = np.squeeze(estimated_data["gt_fluences"])
est_fluence = np.squeeze(estimated_data["est_fluences"])

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

est_fluence[est_fluence <= 1e-10] = 1e-10

num = 1
for phantom_slice_id in range(1, 5):
    path = f"{BASE_PATH}/fold_{str(phantom_slice_id)}/data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua += np.squeeze(estimated_data["est_muas"])
        num += 1

est_mua = est_mua / num

fluence_compensated_signal_bg = signal / gt_fluence
est_fluence_compensated_signal_bg = signal / est_fluence

fluence_gt = (fluence_compensated_signal_bg - FLUENCE_CALIBRATION_INTERCEPT_BG) / FLUENCE_CALIBRATION_SLOPE_BG
fluence_est = (est_fluence_compensated_signal_bg - FLUENCE_CALIBRATION_INTERCEPT_BG) / FLUENCE_CALIBRATION_SLOPE_BG
calibrated_signal = ((signal - CALIBRATION_INTERCEPT) / CALIBRATION_SLOPE)
fluence_gt_mean = np.zeros_like(fluence_gt)
fluence_est_mean = np.zeros_like(fluence_est)
calibrated_signal_mean = np.zeros_like(calibrated_signal)
est_mua_mean = np.zeros_like(est_mua)


for idx in range(len(fluence_gt)):
    seg = gt_segmentation[idx]
    for seg_label in np.unique(seg):
        if seg_label > 1:
            distance = distance_transform_edt((seg == seg_label))
            fluence_gt_mean[idx][(seg == seg_label)] = np.mean(fluence_gt[idx][(seg == seg_label) & (distance < 12)])
            fluence_est_mean[idx][(seg == seg_label)] = np.median(fluence_est[idx][(seg == seg_label) & (distance < 12)])
            calibrated_signal_mean[idx][(seg == seg_label)] = np.mean(calibrated_signal[idx][(seg == seg_label)])
        else:
            fluence_gt_mean[idx][(seg == seg_label)] = np.mean(fluence_gt[idx][(seg == seg_label)])
            fluence_est_mean[idx][(seg == seg_label)] = np.median(fluence_est[idx][(seg == seg_label)])
            calibrated_signal_mean[idx][(seg == seg_label)] = np.percentile(calibrated_signal[idx][(seg == seg_label)], 98)

        est_mua_mean[idx][(seg == seg_label)] = np.mean(est_mua[idx][(seg == seg_label)])

# Profile lines with best-case calibrations for the test set

if not os.path.exists(f"{BASE_PATH}/res_images/"):
    os.makedirs(f"{BASE_PATH}/res_images/")


def plot_result_image(dl_estimate,
                      dl_estimate_mean,
                      phi_estimate,
                      phi_estimate_mean,
                      dl_phi,
                      dl_phi_mean,
                      ground_truth,
                      signal,
                      signal_mean,
                      phantom_slice_id,
                      idx,
                      l_idx,
                      legend=True):
    y0, y1, x0, x1 = points[idx]
    length = int(np.hypot(x1 - x0, y1 - y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    dl_estimate = np.squeeze(dl_estimate)
    ground_truth = np.squeeze(ground_truth)
    signal = np.squeeze(signal)

    # Extract the values along the line
    signal_line = signal[x.astype(int), y.astype(int)]
    fluence_line = phi_estimate[x.astype(int), y.astype(int)]
    est_line = dl_estimate[x.astype(int), y.astype(int)]
    gt_line = ground_truth[x.astype(int), y.astype(int)]
    dl_phi_line = dl_phi[x.astype(int), y.astype(int)]
    signal_mean_line = signal_mean[x.astype(int), y.astype(int)]
    fluence_mean_line = phi_estimate_mean[x.astype(int), y.astype(int)]
    est_mean_line = dl_estimate_mean[x.astype(int), y.astype(int)]
    dl_phi_mean_line = dl_phi_mean[x.astype(int), y.astype(int)]

    ylim_max = np.max(ground_truth) + 0.75
    if np.max(dl_phi_line) > 4 or np.max(fluence_line) > 4:
        ylim_max = 4

    f, axes = plt.subplots(2, 3, figsize=(12, 8))
    ((a0, a2, a4), (a1, a3, a5)) = axes
    im0=a0.imshow(signal, cmap="magma")
    a0.plot([y0, y1], [x0, x1], color=COLOURS[3], linewidth=3, linestyle="dashed")
    a0.axis("off")
    a0.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    divider = make_axes_locatable(a0)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    f.colorbar(im0, cax=cax, orientation='horizontal', label="Cal. Signal [cm$^{-1}$]")

    im2 = a1.imshow(phi_estimate, vmin=0, vmax=np.max(phi_estimate))
    a1.plot([y0, y1], [x0, x1], color=COLOURS[0], linewidth=3, linestyle="dashed")
    a1.axis("off")
    a1.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    divider = make_axes_locatable(a1)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    f.colorbar(im2, cax=cax, orientation='horizontal', label="GT-$\\phi$ [cm$^{-1}$]")

    im1 = a2.imshow(dl_estimate, vmin=0, vmax=ylim_max)
    a2.plot([y0, y1], [x0, x1], color=COLOURS[1], linewidth=3, linestyle="dashed")
    a2.axis("off")
    a2.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    divider = make_axes_locatable(a2)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    f.colorbar(im1, cax=cax, orientation='horizontal', label="DL-$\\hat{\\mu_a}$ [cm$^{-1}$]")

    im3 = a3.imshow(dl_phi, vmin=0, vmax=np.max(phi_estimate))
    a3.plot([y0, y1], [x0, x1], color=COLOURS[2], linewidth=3, linestyle="dashed")
    a3.axis("off")
    a3.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    divider = make_axes_locatable(a3)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    f.colorbar(im3, cax=cax, orientation='horizontal', label="DL-$\\hat{\\phi}$ [cm$^{-1}$]")

    x_axis = np.linspace(0, length, length) * SPACING
    a4.set_title("$\\mu_a$ estimation profiles [cm$^{-1}$]")
    a4.tick_params(direction='in')
    a4.spines.right.set_visible(False)
    a4.spines.top.set_visible(False)
    a4.plot(x_axis, gt_line, color="black", label="GT $\\mu_a$")
    a4.plot(x_axis, signal_line, color=COLOURS[3], linestyle="dashed", label="Cal.")
    a4.plot(x_axis, fluence_line, color=COLOURS[0], linestyle="dashed", label="GT-$\\phi$")
    a4.plot(x_axis, est_line, color=COLOURS[1], linestyle="dashed", label="DL $\\hat{\\mu_a}$")
    a4.plot(x_axis, dl_phi_line, color=COLOURS[2], linestyle="dashed", label="DL $\\hat{\\phi}$")
    a4.set_xlabel("Profile line length [mm]", fontweight="bold")
    if legend:
        a4.legend(frameon=False, prop={"size": 11})

    a5.set_title("Aggregated $\\mu_a$ profiles [cm$^{-1}$]")
    a5.tick_params(direction='in')
    a5.spines.right.set_visible(False)
    a5.spines.top.set_visible(False)
    a5.plot(x_axis, gt_line, color="black", label="GT $\\mu_a$")
    a5.plot(x_axis, signal_mean_line, color=COLOURS[3], linestyle="dashed", label="Cal.")
    a5.plot(x_axis, fluence_mean_line, color=COLOURS[0], linestyle="dashed", label="GT-$\\phi$")
    a5.plot(x_axis, est_mean_line, color=COLOURS[1], linestyle="dashed", label="DL $\\hat{\\mu_a}$")
    a5.plot(x_axis, dl_phi_mean_line, color=COLOURS[2], linestyle="dashed", label="DL $\\hat{\\phi}$")
    a5.set_xlabel("Profile line length [mm]", fontweight="bold")

    axes = axes.flat
    for n, ax in enumerate(axes):
        ax.text(-0.14, 1.02, string.ascii_uppercase[n], transform=ax.transAxes,
                size=30, weight='bold')

    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/res_images/{phantom_slice_id}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{BASE_PATH}/res_images/{phantom_slice_id}.svg", bbox_inches='tight')
    plt.savefig(f"{BASE_PATH}/res_images/{phantom_slice_id}.pdf", bbox_inches='tight')
    plt.close()


print("Plotting results...")
legend = True
for l_idx, (phantom_slice_id, idx) in enumerate(zip(10 + np.asarray(IDXS)*21, IDXS)):
    print(idx, phantom_slice_id)
    plot_result_image(est_mua[phantom_slice_id],
                      est_mua_mean[phantom_slice_id],
                      fluence_gt[phantom_slice_id],
                      fluence_gt_mean[phantom_slice_id],
                      fluence_est[phantom_slice_id],
                      fluence_est_mean[phantom_slice_id],
                      gt_mua[phantom_slice_id],
                      calibrated_signal[phantom_slice_id],
                      calibrated_signal_mean[phantom_slice_id],
                      phantom_slice_id, idx, l_idx,
                      legend=legend)
    legend=True