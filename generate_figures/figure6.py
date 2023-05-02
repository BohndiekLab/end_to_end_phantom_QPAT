from utils.regression import get_mua_regression_line, get_fluence_corrected_regression_line
from patato.core.image_structures.reconstruction_image import Reconstruction
from patato.unmixing.unmixer import SpectralUnmixer, SO2Calculator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.stats import mannwhitneyu
import string
import nrrd
import os

PATH_EXP = r"..\experimental/"
PATH_SIM = r"..\simulation/"
SCAN_INDEX_TO_SHOW = 0

# CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = get_mua_regression_line()

# using the numbers directly for computational speed...
CALIBRATION_SLOPE, CALIBRATION_INTERCEPT = 1484.95, 313.21

COLOURS = [
    "#ccbb44ff",   # YELLOW
    "#4477aaff",  # BLUE
    "#ee6677ff",  # RED
    "#228833ff",  # GREEN
]
SCANS = [1, 15, 17, 21, 23, 3, 5, 7, 9]

WAVELENGTHS = [700, 730, 750, 760, 770, 800, 820, 840, 850, 880]

instance_segmentations = []
segmentations = []

for scan in SCANS:
    nrrd_seg, _ = nrrd.read(fr"C:\final_data_simulation\mouse/Scan_{scan}-labels.nrrd")
    # aorta == 6
    # spine == 5
    # kidney == 4
    # spleen == 3
    # body == 2
    instance_segmentations.append(np.squeeze(nrrd_seg).astype(float))
    _seg = np.squeeze(nrrd_seg).astype(float)
    _seg[_seg <= 1] = -1
    _seg[_seg > 1] = 1
    segmentations.append(_seg)

instance_segmentations = np.asarray(instance_segmentations)
segmentations = np.asarray(segmentations)

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

SPACING = 0.10666666667

# Loading from U-Net trained on experimental data
path = f"{PATH_EXP}/fold_0/mouse_data.npz"
estimated_data = np.load(path)
signal = np.squeeze(estimated_data["gt_inputs"])
print(np.shape(signal))
signal = np.reshape(signal, (-1, len(WAVELENGTHS), 288, 288))
est_mua_exp = np.squeeze(estimated_data["est_muas"])
est_mua_exp = np.reshape(est_mua_exp, (-1, len(WAVELENGTHS), 288, 288))

num = 1
for fold_id in range(1, 5):
    path = f"{PATH_EXP}/fold_{str(fold_id)}/mouse_data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua_exp += np.squeeze(estimated_data["est_muas"].reshape((-1, len(WAVELENGTHS), 288, 288)))
        num += 1
    else:
        print("WARN: Did not find mouse estimate for baseline_", fold_id)
est_mua_exp = est_mua_exp / num

# Loading from U-Net trained on sythetic data
path = f"{PATH_SIM}/fold_0/mouse_data.npz"
estimated_data = np.load(path)
est_mua_sim = np.squeeze(estimated_data["est_muas"])
est_mua_sim = np.reshape(est_mua_sim, (-1, len(WAVELENGTHS), 288, 288))
num = 1
for fold_id in range(1, 5):
    path = f"{PATH_SIM}/fold_{str(fold_id)}/mouse_data.npz"
    if os.path.exists(path):
        estimated_data = np.load(path)
        est_mua_sim += np.squeeze(estimated_data["est_muas"].reshape((-1, len(WAVELENGTHS), 288, 288)))
        num += 1
    else:
        print("WARN: Did not find mouse estimate for baseline_", fold_id)
est_mua_sim = est_mua_sim / num

# Get the calibrated signal
mua_signal = (np.abs(signal)) / CALIBRATION_SLOPE


def define_axis(ax, image, instance_seg, cmap, title, point=None, colour=None, vmin=None, vmax=None, scalebar_color=None):
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
    ax.plot(x_axis, dl_mua_line, COLOURS[1], linestyle="dashed", label="DL-Exp")
    ax.plot(x_axis, dl_phi_line, COLOURS[2], linestyle="dashed", label="DL-Sim")
    if legend:
        ax.legend()

def create_histogram_plot(axis, image_1, image_2, image_3, segmentation, text, _max=1, log=False):
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
    axis.plot(x, d2(x)+1, "--", c=COLOURS[1], label="DL-Exp")
    axis.plot(x, d3(x)+1, "--", c=COLOURS[2], label="DL-Sim")
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
    axis.plot(WAVELENGTHS, s2, c=COLOURS[1], label="DL-Exp")
    axis.plot(WAVELENGTHS, s3, c=COLOURS[2], label="DL-Sim")
    axis.set_xlabel("Wavelength [nm]")
    axis.set_ylabel("Mean $\\mu_a$ in mouse [cm$^{{-1}}$]")


def calc_sO2(image):
    image_shape = np.shape(image)
    tmp_image = np.zeros((image_shape[0], image_shape[1], image_shape[2], image_shape[3], 1))
    tmp_image[:, :, :, :, 0] = image
    wavelengths = np.array(WAVELENGTHS)
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


print(np.shape(mua_signal))
signal_so2 = calc_sO2(mua_signal)
signal_so2[segmentations == -1] = None
mua_so2 = calc_sO2(est_mua_exp)
mua_so2[segmentations == -1] = None
sim_so2 = calc_sO2(est_mua_sim)
sim_so2[segmentations == -1] = None


f = plt.figure(figsize=(6, 14))
gs = f.add_gridspec(4, 2)
gs.update(top=1.0, bottom=0.0, left=0.0, right=1.0)

a1 = f.add_subplot(gs[0, 0])
a5 = f.add_subplot(gs[0, 1])
a2 = f.add_subplot(gs[1, 0])
a6 = f.add_subplot(gs[1, 1])
a3 = f.add_subplot(gs[2, 0])
a7 = f.add_subplot(gs[2, 1])
a9 = f.add_subplot(gs[3, 0])
a11 = f.add_subplot(gs[3, 1])

WL = 5
BORDER = 40
points = [287-2*BORDER-35, 0+35, 287-2*BORDER, 0]

define_axis(a1, mua_signal[SCAN_INDEX_TO_SHOW, WL], instance_segmentations[SCAN_INDEX_TO_SHOW], "magma",
            f"Cal. Signal [cm$^{{-1}}$]", points, COLOURS[0])
define_axis(a2, est_mua_sim[SCAN_INDEX_TO_SHOW, WL], instance_segmentations[SCAN_INDEX_TO_SHOW], "viridis",
            f"DL-Sim [cm$^{{-1}}$]", points, COLOURS[1],
            vmin=0, vmax=None)
define_axis(a3, est_mua_exp[SCAN_INDEX_TO_SHOW, WL], instance_segmentations[SCAN_INDEX_TO_SHOW], "viridis",
            f"DL-Exp [cm$^{{-1}}$]", points, COLOURS[2],
            vmin=0, vmax=None)
# add_lines(a4, mua_signal[WL, 0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           est_mua[WL, 0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           fluence_mua[WL, 0+BORDER:288-BORDER, 0+BORDER:288-BORDER], points, "Line profiles [cm$^{{-1}}$]",
#           lim=[-0.1, 0.75])

define_axis(a5, (signal_so2 * 100)[SCAN_INDEX_TO_SHOW], instance_segmentations[SCAN_INDEX_TO_SHOW], "seismic",
            f"Cal. Signal sO$_2$ [%]", points, COLOURS[0], vmin=0, vmax=100, scalebar_color="black")
define_axis(a6, (sim_so2 * 100)[SCAN_INDEX_TO_SHOW], instance_segmentations[SCAN_INDEX_TO_SHOW], "seismic",
            f"DL-Sim sO$_2$ [%]", points, COLOURS[1], vmin=0, vmax=100, scalebar_color="black")
define_axis(a7, (mua_so2 * 100)[SCAN_INDEX_TO_SHOW], instance_segmentations[SCAN_INDEX_TO_SHOW], "seismic",
            f"DL-Exp sO$_2$ [%]", points, COLOURS[2], vmin=0, vmax=100, scalebar_color="black")
# add_lines(a8, signal_so2[0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           mua_so2[0+BORDER:288-BORDER, 0+BORDER:288-BORDER],
#           phi_so2[0+BORDER:288-BORDER, 0+BORDER:288-BORDER], points, "Line profiles [%]",
#           legend=False)


# ###############################################################
# Ratio between absorption coefficient in aorta and in spine
# ###############################################################

def plot_ratio_end_error(ax, ratios, xpos, colour):
    ax.errorbar(xpos, np.mean(ratios), yerr=np.std(ratios), c=colour, alpha=0.5)
    ax.plot(xpos, np.mean(ratios), "o", c=colour)


def barplot_annotate_brackets(ax, num1, num2, data, center, height, yerr=None, dh=.05, barh=.05,
                              fs=None, maxasterix=4):
    """
    Annotate barplot with p-values.

    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)

quotient_signal = []
quotient_exp = []
quotient_sim = []

for i in range(len(SCANS)):
    quotient_signal.append(np.mean(mua_signal[i, 5][instance_segmentations[i] == 6]) /
                           np.mean(mua_signal[i, 5][instance_segmentations[i] == 5]))
    quotient_exp.append(np.mean(est_mua_exp[i, 5][instance_segmentations[i] == 6]) /
                        np.mean(est_mua_exp[i, 5][instance_segmentations[i] == 5]))
    quotient_sim.append(np.mean(est_mua_sim[i, 5][instance_segmentations[i] == 6]) /
                        np.mean(est_mua_sim[i, 5][instance_segmentations[i] == 5]))

sig_exp_res = mannwhitneyu(quotient_signal, quotient_exp)
p_sig_exp = sig_exp_res.pvalue
sig_sim_res = mannwhitneyu(quotient_signal, quotient_sim)
p_sig_sim = sig_sim_res.pvalue

print(p_sig_exp, p_sig_sim)

plot_ratio_end_error(a9, quotient_signal, 0, COLOURS[0])
plot_ratio_end_error(a9, quotient_sim, 1, COLOURS[1])
plot_ratio_end_error(a9, quotient_exp, 2, COLOURS[2])
barplot_annotate_brackets(a9, 0, 2, p_sig_exp,
                          [0, 1, 2],
                          [np.max(quotient_signal), np.max(quotient_sim), np.max(quotient_exp)], dh=1.2)
barplot_annotate_brackets(a9, 0, 1, p_sig_sim,
                          [0, 1, 2],
                          [np.max(quotient_signal), np.max(quotient_sim), np.max(quotient_exp)], dh=-1.6)
a9.set_xlim(-0.5, 2.5)
a9.set_xticks([0, 1, 2], ["Cal.", "DL-Sim", "DL-Exp"])
a9.spines.right.set_visible(False)
a9.spines.top.set_visible(False)
a9.set_ylabel("Aorta/Spine $\mu_a$-ratio", fontweight="bold")

# ###############################################################
# Aorta oxygenation estimation
# ###############################################################
aorta_sO2_sig = []
aorta_sO2_sim = []
aorta_sO2_exp = []
for i in range(len(SCANS)):
    aorta_sO2_sig.append(np.mean(signal_so2[i][instance_segmentations[i] == 6] * 100))
    aorta_sO2_sim.append(np.mean(sim_so2[i][instance_segmentations[i] == 6] * 100))
    aorta_sO2_exp.append(np.mean(mua_so2[i][instance_segmentations[i] == 6] * 100))

aorta_sO2_sig = np.asarray(aorta_sO2_sig)
aorta_sO2_sim = np.asarray(aorta_sO2_sim)
aorta_sO2_exp = np.asarray(aorta_sO2_exp)

sig_exp_so2 = mannwhitneyu(aorta_sO2_sig, aorta_sO2_exp)
p_sig_exp_so2 = sig_exp_so2.pvalue
sig_sim_so2 = mannwhitneyu(aorta_sO2_sig, aorta_sO2_sim)
p_sig_sim_so2 = sig_sim_so2.pvalue

print("exp")
print(np.median(np.abs(aorta_sO2_exp-aorta_sO2_sig)), np.std(aorta_sO2_exp-aorta_sO2_sig))
print(p_sig_exp_so2)

print("sim")
print(np.median(np.abs(aorta_sO2_sim-aorta_sO2_sig)), np.std(aorta_sO2_sim-aorta_sO2_sig))
print(p_sig_sim_so2)

bp2 = a11.boxplot([aorta_sO2_sig, aorta_sO2_sim, aorta_sO2_exp],
                  showfliers=False, widths=0.8)

barplot_annotate_brackets(a11, 0, 1, p_sig_sim_so2,
                          [1, 2, 3],
                          [np.max(aorta_sO2_sig), np.max(aorta_sO2_sim), np.max(aorta_sO2_exp)], dh=2)
barplot_annotate_brackets(a11, 0, 2, p_sig_exp_so2,
                          [1, 2, 3],
                          [np.max(aorta_sO2_sig), np.max(aorta_sO2_sim), np.max(aorta_sO2_exp)], dh=2)

for idx, median in enumerate(bp2['medians']):
    median.set_color(COLOURS[idx])
parts = a11.violinplot([aorta_sO2_sig, aorta_sO2_sim, aorta_sO2_exp], widths=0.8, showextrema=False)
for idx, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COLOURS[idx])
    pc.set_edgecolor(COLOURS[idx])
    pc.set_alpha(0.4)
a11.set_xticks([1, 2, 3], ["Cal.", "DL-Sim", "DL-Exp"])
a11.spines.right.set_visible(False)
a11.spines.top.set_visible(False)
a11.set_ylabel("Aorta sO$_2$ [%]", fontweight="bold")
a11.fill_between([0.5, 3.5], 94, 98, color="green", alpha=0.25)
a11.set_ylim(None, 104)
a11.hlines(96, xmin=0.5, xmax=3.5, color="green")

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


a1.text(0.05, 0.85, "A", transform=a1.transAxes,
        size=30, weight='bold', color="white")
a5.text(0.05, 0.85, "B", transform=a5.transAxes,
        size=30, weight='bold')
a2.text(0.05, 0.85, "C", transform=a2.transAxes,
        size=30, weight='bold', color="white")
a6.text(0.05, 0.85, "D", transform=a6.transAxes,
        size=30, weight='bold')
a3.text(0.05, 0.85, "E", transform=a3.transAxes,
        size=30, weight='bold', color="white")
a7.text(0.05, 0.85, "F", transform=a7.transAxes,
        size=30, weight='bold')
a9.text(0.02, 0.9, "G", transform=a9.transAxes,
        size=30, weight='bold')
a11.text(0.02, 0.9, "H", transform=a11.transAxes,
        size=30, weight='bold')

a1.text(0.12, 0.52, "SPLEEN", transform=a1.transAxes,
        size=10, weight='bold', color="white")
a1.text(0.3, 0.15, "KIDNEY", transform=a1.transAxes,
        size=10, weight='bold', color="white")
a1.text(0.65, 0.65, "KIDNEY", transform=a1.transAxes,
        size=10, weight='bold', color="white")
a1.text(0.45, 0.44, "AORTA", transform=a1.transAxes,
        size=10, weight='bold', color="white")
a1.text(0.6, 0.14, "SPINE", transform=a1.transAxes,
        size=10, weight='bold', color="white")

plt.savefig(f"../figures/figure6.png", bbox_inches='tight', dpi=300)
plt.savefig(f"../figures/figure6.svg", bbox_inches='tight')
plt.savefig(f"../figures/figure6.pdf", bbox_inches='tight')
plt.close()