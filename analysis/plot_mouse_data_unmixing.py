from utils.regression import get_mua_regression_line, get_fluence_corrected_regression_line
from patato.core.image_structures.reconstruction_image import Reconstruction
from patato.unmixing.unmixer import SpectralUnmixer, SO2Calculator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.stats import gaussian_kde
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

# This is a training set-optimised property to discard pixels too deep inside the structures
DISTANCE_THRESHOLD = 12

CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = get_mua_regression_line(BASE_PATH)
FLUENCE_CALIBRATION_SLOPE_BG, FLUENCE_CALIBRATION_INTERCEPT_BG = get_fluence_corrected_regression_line(
    DISTANCE_THRESHOLD)

COLOURS = [
    "#ccbb44ff",   # YELLOW
    "#ee6677ff",  # RED
    "#4477aaff",  # BLUE
    "#228833ff",  # GREEN
]

WAVELENGTHS = [700, 730, 750, 760, 770, 800, 820, 840, 850, 880]

nrrd_seg, _ = nrrd.read(r"C:\final_data_simulation\mouse/Scan_1-labels.nrrd")
# aorta == 6
# spine == 5
# kidney == 4
# spleen == 3
# body == 2
instance_seg = np.squeeze(nrrd_seg).astype(float)
segmentation = np.squeeze(nrrd_seg).astype(float)
segmentation[segmentation <= 1] = -1
segmentation[segmentation > 1] = 1

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

SPACING = 0.10666666667
path = f"{BASE_PATH}/fold_0/mouse_data.npz"
estimated_data = np.load(path)
signal = np.squeeze(estimated_data["gt_inputs"])
est_mua = np.squeeze(estimated_data["est_muas"])
est_fluence = np.squeeze(estimated_data["est_fluences"])

num = 1
for fold_id in range(1, 5):
    path = f"{BASE_PATH}/fold_{str(fold_id)}/mouse_data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua += np.squeeze(estimated_data["est_muas"])
        est_fluence += np.squeeze(estimated_data["est_fluences"])
        num += 1
    else:
        print("WARN: Did not find mouse estimate for baseline_", fold_id)

est_mua = est_mua / num
est_fluence = est_fluence / num


mua_signal = (np.abs(signal)) / CALIBRATION_SLOPE
fluence_mua = (np.abs(signal) / est_fluence) / FLUENCE_CALIBRATION_SLOPE_BG


def define_axis(ax, image, cmap, title, point=None, colour=None, vmin=None, vmax=None, scalebar_color=None):
    im = ax.imshow(image[0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    if scalebar_color is None:
        scalebar_color = "white"
    ax.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color=scalebar_color, box_alpha=0))
    ax.contour((instance_seg == 6)[0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER])
    ax.contour((instance_seg == 5)[0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER])
    ax.contour((instance_seg == 4)[0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER])
    ax.contour((instance_seg == 3)[0 + BORDER:288 - BORDER, 0 + BORDER:288 - BORDER])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='horizontal', label=title)
    # ax.set_title(title)
    if point is not None:
        y0, y1, x0, x1 = point
        # ax.plot([y0, y1], [x0, x1], colour, linewidth=3, linestyle="dashed")

def add_lines(ax, sig, dl_mua, dl_phi, point, title=None, lim=None, legend=True):
    if title is None:
        title = "Line profiles [cm$^{{-1}}$]"
    if lim is None:
        lim = [-0.1, 1.1]
    ax.tick_params(direction='in')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_ylabel(title)
    ax.set_xlabel("Position on line [mm]")
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
    if legend:
        ax.legend()

def create_histogram_plot(axis, image_1, image_2, image_3, text, _max=1, log=False):
    image_1 = np.reshape(image_1[segmentation == 1], (-1, ))
    image_2 = np.reshape(image_2[segmentation == 1], (-1,))
    image_3 = np.reshape(image_3[segmentation == 1], (-1,))
    axis.tick_params(direction='in')
    axis.spines.right.set_visible(False)
    axis.spines.top.set_visible(False)
    d1 = gaussian_kde(image_1)
    d2 = gaussian_kde(image_2)
    d3 = gaussian_kde(image_3)
    x = np.linspace(0, _max, 100)
    if log:
        x = np.log10(x+1)
    axis.plot(x, d1(x)+1, "--", c=COLOURS[0], label="Cal.")
    axis.plot(x, d2(x)+1, "--", c=COLOURS[1], label="DL-$\\hat{\\mu_a}$")
    axis.plot(x, d3(x)+1, "--", c=COLOURS[2], label="DL-$\\phi$")
    axis.set_ylabel("Norm. probability density")
    axis.set_xlabel(text)


def plot_mean_spectrum(axis, image_1, image_2, image_3):
    s1 = np.mean(image_1, axis=(1, 2))
    s2 = np.mean(image_2, axis=(1, 2))
    s3 = np.mean(image_3, axis=(1, 2))
    axis.tick_params(direction='in')
    axis.spines.right.set_visible(False)
    axis.spines.top.set_visible(False)
    axis.plot(WAVELENGTHS, s1, c=COLOURS[0], label="Cal.")
    axis.plot(WAVELENGTHS, s2, c=COLOURS[1], label="DL-$\\hat{\\mu_a}$")
    axis.plot(WAVELENGTHS, s3, c=COLOURS[2], label="DL-$\\phi$")
    axis.set_xlabel("Wavelength [nm]")
    axis.set_ylabel("Mean $\\mu_a$ in mouse [cm$^{{-1}}$]")


def calc_sO2(image):
    image_shape = np.shape(image)
    tmp_image = np.zeros((1, image_shape[0], image_shape[1], image_shape[2], 1))
    tmp_image[0, :, :, :, 0] = image
    wavelengths = np.array(WAVELENGTHS)
    print(np.shape(tmp_image))
    print(np.shape(wavelengths))
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


signal_so2 = calc_sO2(mua_signal[:len(WAVELENGTHS), :, :])
signal_so2[segmentation==-1] = None
mua_so2 = calc_sO2(est_mua[:len(WAVELENGTHS), :, :])
mua_so2[segmentation==-1] = None
phi_so2 = calc_sO2(fluence_mua[:len(WAVELENGTHS), :, :])
phi_so2[segmentation==-1] = None

fig, axes = plt.subplots(2, 4, figsize=(12, 7.3))
((a1, a2, a3, a9), (a5, a6, a7, a11)) = axes

WL = 5
BORDER = 40
points = [287-2*BORDER-35, 0+35, 287-2*BORDER, 0]

define_axis(a1, mua_signal[WL], "magma",
            f"Cal. Signal [cm$^{{-1}}$]", points, COLOURS[0])
define_axis(a2, est_mua[WL], "viridis",
            f"DL-$\hat{{\mu_a}}$ [cm$^{{-1}}$]", points, COLOURS[1],
            vmin=0, vmax=None)
define_axis(a3, fluence_mua[WL], "viridis",
            f"DL-$\hat{{\phi}}$ [cm$^{{-1}}$]", points, COLOURS[2],
            vmin=0, vmax=None)
# add_lines(a4, mua_signal[WL, 0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           est_mua[WL, 0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           fluence_mua[WL, 0+BORDER:288-BORDER, 0+BORDER:288-BORDER], points, "Line profiles [cm$^{{-1}}$]",
#           lim=[-0.1, 0.75])

define_axis(a5, signal_so2 * 100, "seismic",
            f"Cal. Signal sO$_2$ [%]", points, COLOURS[0], vmin=0, vmax=100, scalebar_color="black")
define_axis(a6, mua_so2 * 100, "seismic",
            f"DL-$\hat{{\mu_a}}$ sO$_2$ [%]", points, COLOURS[1], vmin=0, vmax=100, scalebar_color="black")
define_axis(a7, phi_so2 * 100, "seismic",
            f"DL-$\hat{{\phi}}$ sO$_2$ [%]", points, COLOURS[2], vmin=0, vmax=100, scalebar_color="black")
# add_lines(a8, signal_so2[0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           mua_so2[0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           phi_so2[0+BORDER:288-BORDER, 0+BORDER:288-BORDER], points, "Line profiles [%]",
#           legend=False)


# ###############################################################
# Ratio between absorption coefficient in aorta and in spine
# ###############################################################

def plot_ratio_end_error(ax, a, b, xpos, colour):
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    error = mean_a/mean_b * np.sqrt((std_a/mean_a)**2 + (std_b/mean_b)**2)
    ax.errorbar(xpos, mean_a/mean_b, yerr=error, c=colour, alpha=0.5)
    ax.plot(xpos, mean_a/mean_b, "o", c=colour)


plot_ratio_end_error(a9, mua_signal[5][instance_seg==6], mua_signal[5][instance_seg==5], 0, COLOURS[0])
plot_ratio_end_error(a9, est_mua[5][instance_seg==6], est_mua[5][instance_seg==5], 1, COLOURS[1])
plot_ratio_end_error(a9, fluence_mua[5][instance_seg==6], fluence_mua[5][instance_seg==5], 2, COLOURS[2])
a9.set_xlim(-0.5, 2.5)
a9.set_xticks([0, 1, 2], ["Cal.", "DL-$\hat{{\mu_a}}$", "DL-$\hat{{\phi}}$"])
a9.spines.right.set_visible(False)
a9.spines.top.set_visible(False)
a9.set_ylabel("Aorta/Spine $\mu_a$-ratio", fontweight="bold")

# ###############################################################
# Multiwavelength spectrum of aorta
# ###############################################################

# a10.plot(WAVELENGTHS, np.mean(mua_signal[:, instance_seg==6], axis=1), c=COLOURS[0])
# a10.plot(WAVELENGTHS, np.mean(est_mua[:, instance_seg==6], axis=1), c=COLOURS[1])
# a10.plot(WAVELENGTHS, np.mean(fluence_mua[:, instance_seg==6], axis=1), c=COLOURS[2])
# a10.spines.right.set_visible(False)
# a10.spines.top.set_visible(False)
# a10.set_xlabel("Wavelength [nm]")
# a10.set_ylabel("$\\mu_a$ [cm$^{-1}$]")

# ###############################################################
# Aorta oxygenation estimation
# ###############################################################
aorta_sO2 = [signal_so2[instance_seg==6] * 100, mua_so2[instance_seg==6] * 100, phi_so2[instance_seg==6] * 100]
bp2 = a11.boxplot(aorta_sO2,
                 showfliers=False, widths=0.8)
for idx, median in enumerate(bp2['medians']):
    median.set_color(COLOURS[idx])
parts = a11.violinplot(aorta_sO2, widths=0.8, showextrema=False)
for idx, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COLOURS[idx])
    pc.set_edgecolor(COLOURS[idx])
    pc.set_alpha(0.4)
a11.set_xticks([1, 2, 3], ["Cal.", "DL-$\hat{{\mu_a}}$", "DL-$\hat{{\phi}}$"])
a11.spines.right.set_visible(False)
a11.spines.top.set_visible(False)
a11.set_ylabel("Aorta sO$_2$ [%]", fontweight="bold")

# create_histogram_plot(a9, mua_signal[0, :, :], est_mua[0, :, :], fluence_mua[0, :, :],
#                       f"log($\\hat{{\\mu_a}} + 1$) at {WAVELENGTHS[0]}nm [a.u.]", log=True, _max=2)
# create_histogram_plot(a10, mua_signal[9, :, :], est_mua[9, :, :], fluence_mua[9, :, :],
#                       f"log($\\hat{{\\mu_a}} + 1$) at {WAVELENGTHS[9]}nm [a.u.]", log=True, _max=2)
# signal_so2[signal_so2 < 0] = 0
# signal_so2[signal_so2 > 1] = 1
# mua_so2[mua_so2 < 0] = 0
# mua_so2[mua_so2 > 1] = 1
# phi_so2[phi_so2 < 0] = 0
# phi_so2[phi_so2 > 1] = 1
# create_histogram_plot(a11, signal_so2 * 100, mua_so2 * 100, phi_so2 * 100,
#                       f"Estimated $sO_2$ [%]", _max=100)
# plot_mean_spectrum(a12, mua_signal[0:, :, :], est_mua[0:, :, :], fluence_mua[0:, :, :])

axes = axes.flat
for n, ax in enumerate(axes):
    ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
            size=30, weight='bold')

plt.tight_layout()
plt.savefig(f"{BASE_PATH}/figures/mouse_image.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{BASE_PATH}/figures/mouse_image.svg", bbox_inches='tight')
plt.savefig(f"{BASE_PATH}/figures/mouse_image.pdf", bbox_inches='tight')
plt.close()