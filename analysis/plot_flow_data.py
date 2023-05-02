from utils.regression import get_mua_regression_line, get_fluence_corrected_regression_line
from patato.core.image_structures.reconstruction_image import Reconstruction
from patato.unmixing.unmixer import SpectralUnmixer, SO2Calculator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import string
import nrrd
import os

EXPERIMENTAL = False

if EXPERIMENTAL:
    BASE_PATH = r"C:\palpaitine\02-end-to-end-deep-learning\mua_phi/experimental/"
else:
    BASE_PATH = r"C:\palpaitine\02-end-to-end-deep-learning\mua_phi/simulation/"

COLOURS = [
    "#ccbb44ff",   # YELLOW
    "#ee6677ff",  # RED
    "#4477aaff",  # BLUE
    "#228833ff",  # GREEN
]

WAVELENGTHS = [660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960]

WL_START = 2
WL_END = 13

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)


SPACING = 0.10666666667
path = f"{BASE_PATH}/fold_0/flow_data.npz"

# This is a training set-optimised property to discard pixels too deep inside the structures
DISTANCE_THRESHOLD = 12

CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = get_mua_regression_line(BASE_PATH)
FLUENCE_CALIBRATION_SLOPE_BG, FLUENCE_CALIBRATION_INTERCEPT_BG = get_fluence_corrected_regression_line(DISTANCE_THRESHOLD)

estimated_data = np.load(path)
signal = np.squeeze(estimated_data["gt_inputs"])
est_mua = np.squeeze(estimated_data["est_muas"])
est_fluence = np.squeeze(estimated_data["est_fluences"])

segmentation, _ = nrrd.read(r"C:\final_data_simulation\flow/Scan13_0-labels.nrrd")
segmentation = np.squeeze(segmentation).astype(float)
seg_background = np.zeros_like(segmentation)
seg_melanin = np.zeros_like(segmentation)
seg_phantom = np.zeros_like(segmentation)
seg_blood = np.zeros_like(segmentation)
seg_background[segmentation == 1] = 1
seg_melanin[segmentation == 2] = 1
seg_phantom[segmentation == 3] = 1
seg_blood[segmentation == 4] = 1

num = 1
for fold_id in range(1, 5):
    path = f"{BASE_PATH}/fold_{str(fold_id)}/flow_data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua += np.squeeze(estimated_data["est_muas"])
        est_fluence += np.squeeze(estimated_data["est_fluences"])
        num += 1
    else:
        print("WARN: Did not find mouse estimate for baseline_", fold_id)

est_mua = est_mua / num
est_fluence = est_fluence / num

mua_signal = np.abs(signal) / CALIBRATION_SLOPE
fluence_mua = (np.abs(signal) / est_fluence) / FLUENCE_CALIBRATION_SLOPE_BG

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar


def define_axis(ax, im, title, point=None, colour=None):
    ax.axis("off")
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='horizontal', label=title)
    # ax.set_title(title)
    if point is not None:
        y0, y1, x0, x1 = point
        ax.plot([y0, y1], [x0, x1], colour, linewidth=3, linestyle="dashed")


def add_lines(ax, sig, dl_mua, dl_phi, point, title=None, lim=None):
    if title is None:
        title = "Line profiles [cm$^{{-1}}$]"
    if lim is None:
        lim = [-0.1, 1.1]
    ax.tick_params(direction='in')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_ylabel(title, fontweight="bold")
    ax.set_xlabel("Position on line [mm]", fontweight="bold")
    ax.set_ylim(lim)
    y0, y1, x0, x1 = point
    length = int(np.hypot(x1 - x0, y1 - y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    x_axis = np.linspace(0, length, length) * SPACING

    # Extract the values along the line
    sig_line = sig[x.astype(int), y.astype(int)]
    dl_mua_line = dl_mua[x.astype(int), y.astype(int)]
    dl_phi_line = dl_phi[x.astype(int), y.astype(int)]

    ax.plot(x_axis, sig_line, COLOURS[0], linestyle="dashed", label="Cal.")
    ax.plot(x_axis, dl_mua_line, COLOURS[1], linestyle="dashed", label="DL-$\\hat{\\mu_a}$")
    ax.plot(x_axis, dl_phi_line, COLOURS[2], linestyle="dashed", label="DL-$\\phi$")


def calc_sO2(image):
    image_shape = np.shape(image)
    tmp_image = np.zeros((1, image_shape[0], image_shape[1], image_shape[2], 1))
    tmp_image[0, :, :, :, 0] = image
    wavelengths = np.array(WAVELENGTHS[WL_START:WL_END])
    r = Reconstruction(tmp_image, wavelengths,
                       field_of_view=(1, 1, 1))  # field of view is the width of the image along x, y, z
    r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_X"] = 1
    r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_Y"] = 1
    r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_Z"] = 1
    um = SpectralUnmixer(["Hb", "HbO2"], r.wavelengths)
    so = SO2Calculator()
    um, _, _ = um.run(r, None)
    so2, _, _ = so.run(um, None)

    return np.squeeze(so2.values)


def plot_mean_spectrum(axis, image_1, image_2, image_3):
    s1 = np.mean(image_1, axis=1)
    s2 = np.mean(image_2, axis=1)
    s3 = np.mean(image_3, axis=1)
    axis.tick_params(direction='in')
    axis.spines.right.set_visible(False)
    axis.spines.top.set_visible(False)
    axis.plot(WAVELENGTHS[WL_START:WL_END], s1, c=COLOURS[0], label="Cal.")
    axis.plot(WAVELENGTHS[WL_START:WL_END], s2, c=COLOURS[1], label="DL-$\\hat{\\mu_a}$")
    axis.plot(WAVELENGTHS[WL_START:WL_END], s3, c=COLOURS[2], label="DL-$\\phi$")
    axis.set_xlabel("Wavelength [nm]", fontweight="bold")
    axis.set_ylabel("Blood absorption [cm$^{{-1}}$]", fontweight="bold")


signal_sO2 = [None] * 5
mua_so2 = [None] * 5
phi_so2 = [None] * 5
for _i, idx in enumerate(range(5)):
    signal_sO2[_i] = calc_sO2(mua_signal[idx * len(WAVELENGTHS):(idx+1) * len(WAVELENGTHS), :, :][WL_START:WL_END])[seg_blood == 1]
    mua_so2[_i] = calc_sO2(est_mua[idx * len(WAVELENGTHS):(idx + 1) * len(WAVELENGTHS), :, :][WL_START:WL_END])[seg_blood == 1]
    phi_so2[_i] = calc_sO2(fluence_mua[idx * len(WAVELENGTHS):(idx + 1) * len(WAVELENGTHS), :, :][WL_START:WL_END])[seg_blood == 1]


fig, axes = plt.subplots(2, 4, figsize=(12, 7.3))
((a1, a2, a3, a4), (a5, a6, a7, a8)) = axes

WL = 7
BORDER = 40
points = [0, 287 - 2 * BORDER, 145 - BORDER, 145 - BORDER]

define_axis(a1, a1.imshow(mua_signal[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], cmap="magma"),
            f"Cal. Signal [cm$^{{-1}}$]", points, COLOURS[0])
define_axis(a2, a2.imshow(est_mua[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], vmin=0, vmax=None),
            f"DL-$\hat{{\mu_a}}$ [cm$^{{-1}}$]", points, COLOURS[1])
define_axis(a3, a3.imshow(fluence_mua[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], vmin=0, vmax=None),
            f"DL-$\hat{{\phi}}$ [cm$^{{-1}}$]", points, COLOURS[2])
add_lines(a4, mua_signal[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER],
          est_mua[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER],
          fluence_mua[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], points, "Line profiles [cm$^{{-1}}$]",
          lim=[-0.1, 1.7])
time_min = [0.1, 5.7, 14.0, 27.8, 41.7]
sig_sO2 = np.mean(signal_sO2, axis=1)*100
a5.plot(time_min, sig_sO2, c=COLOURS[0], label="Cal.")
a5.plot(time_min, np.mean(mua_so2, axis=1)*100, c=COLOURS[1], label="DL-$\\hat{\\mu_a}$")
a5.plot(time_min, np.mean(phi_so2, axis=1)*100, c=COLOURS[2], label="DL-$\\phi$")
a5.annotate("F", (time_min[0] - 1, sig_sO2[0] + 3))
a5.plot((time_min[0], time_min[0]), (0, sig_sO2[0]+2), "--", c="black", alpha=0.5)
a5.annotate("G", (time_min[2] - 1.25, sig_sO2[2] + 3))
a5.plot((time_min[2], time_min[2]), (0, sig_sO2[2]+2), "--", c="black", alpha=0.5)
a5.annotate("H", (time_min[4] - 1.25, sig_sO2[4] + 3))
a5.plot((time_min[4], time_min[4]), (0, sig_sO2[4]+2), "--", c="black", alpha=0.5)
a5.tick_params(direction='in')
a5.spines.right.set_visible(False)
a5.spines.top.set_visible(False)
a5.set_xlabel("Imaging time [min]", fontweight="bold")
a5.set_ylabel("Mean estimated $sO_2$ [%]", fontweight="bold")

plot_mean_spectrum(a6, mua_signal[0:len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1],
                   est_mua[0:len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1],
                   fluence_mua[0:len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1])
plot_mean_spectrum(a7, mua_signal[2*len(WAVELENGTHS):3*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1],
                   est_mua[2*len(WAVELENGTHS):3*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1],
                   fluence_mua[2*len(WAVELENGTHS):3*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1])
plot_mean_spectrum(a8, mua_signal[4*len(WAVELENGTHS):5*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1],
                   est_mua[4*len(WAVELENGTHS):5*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1],
                   fluence_mua[4*len(WAVELENGTHS):5*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1])
a8.legend(fontsize=10)

axes = axes.flat
for n, ax in enumerate(axes):
    ax.text(-0.15, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
            size=30, weight='bold')

plt.tight_layout()
plt.savefig(f"{BASE_PATH}/figures/flow_phantom.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{BASE_PATH}/figures/flow_phantom.svg", bbox_inches='tight')
plt.savefig(f"{BASE_PATH}/figures/flow_phantom.pdf", bbox_inches='tight')
plt.close()
