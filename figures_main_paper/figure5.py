from utils.regression import get_mua_regression_line, get_fluence_corrected_regression_line
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from patato.core.image_structures.reconstruction_image import Reconstruction
from patato.unmixing.unmixer import SpectralUnmixer, SO2Calculator
import matplotlib.pyplot as plt
from data_path import DATA_PATH
import numpy as np
import simpa as sp
import matplotlib
import nrrd
import os

hb = sp.AbsorptionSpectrumLibrary().get_spectrum_by_name("Deoxyhemoglobin")
hbo2 = sp.AbsorptionSpectrumLibrary().get_spectrum_by_name("Oxyhemoglobin")
wavelength_range = np.asarray(range(700, 901, 10))
hb_spectrum = []
hbo2_spectrum = []
for i in wavelength_range:
    hb_spectrum.append(hb.get_value_for_wavelength(i))
    hbo2_spectrum.append(hbo2.get_value_for_wavelength(i))

hb_spectrum = np.asarray(hb_spectrum)
hbo2_spectrum = np.asarray(hbo2_spectrum)

PATH_EXP = rf"{DATA_PATH}/model_weights_experiment/"
PATH_SIM = rf"{DATA_PATH}/model_weights_simulation/"

if not os.path.exists("../figures/res_images/"):
    os.makedirs("../figures/res_images/")

COLOURS = [
    "#ccbb44ff",   # YELLOW
    "#ee6677ff",  # RED
    "#4477aaff",  # BLUE
    "#228833ff",  # GREEN
]

WAVELENGTHS = [660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960]

TIMES = [34.8, 55.7, 76.6, 97.7]
OXYGENATION = [0.9952168555996834, 0.9903691873483731, 0.7002687502408534, 0.06030676906516432]

WL_START = 2
WL_END = 13

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)


SPACING = 0.10666666667
path = f"{PATH_EXP}/fold_0/flow_data.npz"

# This is a training set-optimised property to discard pixels too deep inside the structures
DISTANCE_THRESHOLD = 12

# CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = get_mua_regression_line()

# using the numbers directly for computational speed...
CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = 1484.95, 313.21

estimated_data = np.load(path)
signal = np.squeeze(estimated_data["gt_inputs"])
est_mua_exp = np.squeeze(estimated_data["est_muas"])

segmentation, _ = nrrd.read(rf"{DATA_PATH}\flow/Scan13_0-labels.nrrd")
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
    path = f"{PATH_EXP}/fold_{str(fold_id)}/flow_data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua_exp += np.squeeze(estimated_data["est_muas"])
        num += 1
    else:
        print("WARN: Did not find mouse estimate for baseline_", fold_id)

est_mua_exp = est_mua_exp / num

path = f"{PATH_SIM}/fold_0/flow_data.npz"
estimated_data = np.load(path)
signal = np.squeeze(estimated_data["gt_inputs"])
est_mua_sim = np.squeeze(estimated_data["est_muas"])
num = 1
for fold_id in range(1, 5):
    path = f"{PATH_SIM}/fold_{str(fold_id)}/flow_data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua_sim += np.squeeze(estimated_data["est_muas"])
        num += 1
    else:
        print("WARN: Did not find mouse estimate for baseline_", fold_id)
est_mua_sim = est_mua_sim / num

mua_signal = np.abs(signal) / CALIBRATION_SLOPE

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
    ax.plot(x_axis, dl_mua_line, COLOURS[1], linestyle="dashed", label="DL-Exp")
    ax.plot(x_axis, dl_phi_line, COLOURS[2], linestyle="dashed", label="DL-Sim")


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


signal_sO2 = [None] * len(TIMES)
mua_so2_exp = [None] * len(TIMES)
mua_so2_sim = [None] * len(TIMES)
for _i, idx in enumerate(range(2, 2+len(TIMES))):
    signal_sO2[_i] = calc_sO2(mua_signal[idx * len(WAVELENGTHS):(idx+1) * len(WAVELENGTHS), :, :][WL_START:WL_END])[seg_blood == 1]
    mua_so2_exp[_i] = calc_sO2(est_mua_exp[idx * len(WAVELENGTHS):(idx + 1) * len(WAVELENGTHS), :, :][WL_START:WL_END])[seg_blood == 1]
    mua_so2_sim[_i] = calc_sO2(est_mua_sim[idx * len(WAVELENGTHS):(idx + 1) * len(WAVELENGTHS), :, :][WL_START:WL_END])[seg_blood == 1]

f = plt.figure(figsize=(6, 11))
gs = f.add_gridspec(3, 4)
# a0 = f.add_subplot(gs[0, 0:2])
# a1 = f.add_subplot(gs[0, 2:4])
#
# a2 = f.add_subplot(gs[1, 0:2])
a4 = f.add_subplot(gs[0, :])

a3 = f.add_subplot(gs[1, :])

a5 = f.add_subplot(gs[2, 0:2])
a6 = f.add_subplot(gs[2, 2:4])

WL = 7
BORDER = 40
points = [0, 287 - 2 * BORDER, 145 - BORDER, 145 - BORDER]

# define_axis(a0, a0.imshow(mua_signal[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], cmap="magma"),
#             f"Cal. Signal [cm$^{{-1}}$]", points, COLOURS[0])
# define_axis(a1, a1.imshow(est_mua_exp[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], vmin=0, vmax=None),
#             f"DL-Exp [cm$^{{-1}}$]", points, COLOURS[1])
# define_axis(a2, a2.imshow(est_mua_sim[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], vmin=0, vmax=None),
#             f"DL-Sim [cm$^{{-1}}$]", points, COLOURS[2])
add_lines(a4, mua_signal[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER],
          est_mua_exp[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER],
          est_mua_sim[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], points, "Line profiles [cm$^{{-1}}$]",
          lim=[-0.1, 2.7])
a4.legend(fontsize=10)

imagebox = OffsetImage(mua_signal[WL, 0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], zoom=0.5)
ab = AnnotationBbox(imagebox, (5, 1.5), frameon=False, zorder=1)
a4.add_artist(ab)
a4.hlines(1.475, 1, 9, colors="white", linestyles="dashed", zorder=3)
a4.text(1.4, 2.35, "PA reference\nimage @800nm")
a4.text(-0.1, 1.02, "A", transform=a4.transAxes, size=30, weight='bold')

sig_sO2 = np.mean(signal_sO2, axis=1)*100
a3.plot(TIMES, np.asarray(OXYGENATION) * 100, c="green", linestyle="dashed", label="pO$_2$ reference")
a3.plot(TIMES, sig_sO2, c=COLOURS[0], label="Cal.")
a3.plot(TIMES, np.mean(mua_so2_exp, axis=1) * 100, c=COLOURS[1], label="DL-Exp")
a3.plot(TIMES, np.mean(mua_so2_sim, axis=1) * 100, c=COLOURS[2], label="DL-Sim")
a3.legend()
a3.annotate("C", (TIMES[0] - 1, 105))
a3.plot((TIMES[0], TIMES[0]), (0, 100), "--", c="black", alpha=0.5)
# a3.annotate("G", (time_min[2] - 1.25, sig_sO2[2] + 3))
# a3.plot((time_min[2], time_min[2]), (0, sig_sO2[2] + 2), "--", c="black", alpha=0.5)
a3.annotate("D", (TIMES[-1] - 1.25, sig_sO2[-1] + 10))
a3.plot((TIMES[-1], TIMES[-1]), (0, sig_sO2[-1] + 8), "--", c="black", alpha=0.5)
a3.tick_params(direction='in')
a3.spines.right.set_visible(False)
a3.spines.top.set_visible(False)
a3.set_xlabel("Imaging time [min]", fontweight="bold")
a3.set_ylabel("Mean estimated $sO_2$ [%]", fontweight="bold")
a3.text(-0.1, 1.02, "B", transform=a3.transAxes, size=30, weight='bold')

plot_mean_spectrum(a5, mua_signal[:len(WAVELENGTHS)][WL_START:WL_END, seg_blood == 1],
                   est_mua_exp[:len(WAVELENGTHS)][WL_START:WL_END, seg_blood == 1],
                   est_mua_sim[:len(WAVELENGTHS)][WL_START:WL_END, seg_blood == 1])
# plot_mean_spectrum(a6, mua_signal[2*len(WAVELENGTHS):3*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1],
#                    est_mua_exp[2 * len(WAVELENGTHS):3 * len(WAVELENGTHS)][WL_START:WL_END, seg_blood == 1],
#                    est_mua_sim[2*len(WAVELENGTHS):3*len(WAVELENGTHS)][WL_START:WL_END, seg_blood==1])
plot_mean_spectrum(a6, mua_signal[-len(WAVELENGTHS):][WL_START:WL_END, seg_blood == 1],
                   est_mua_exp[-len(WAVELENGTHS):][WL_START:WL_END, seg_blood == 1],
                   est_mua_sim[-len(WAVELENGTHS):][WL_START:WL_END, seg_blood == 1])

a5.text(-0.3, 1.02, "C", transform=a5.transAxes, size=30, weight='bold')
a6.text(-0.3, 1.02, "D", transform=a6.transAxes, size=30, weight='bold')

a5.plot(wavelength_range, hbo2_spectrum/3, color="black", linestyle="dashed")
a5.text(0.5, 0.75, "lit. ref. / 3", transform=a5.transAxes, size=10, weight='bold')
a6.plot(wavelength_range, hb_spectrum/3, color="black", linestyle="dashed")
a6.text(0.4, 0.75, "lit. ref. / 3", transform=a6.transAxes, size=10, weight='bold')


plt.tight_layout()
plt.savefig(f"../figures/figure5.png", bbox_inches='tight', dpi=300)
plt.savefig(f"../figures/figure5.svg", bbox_inches='tight')
plt.savefig(f"../figures/figure5.pdf", bbox_inches='tight')
plt.close()
