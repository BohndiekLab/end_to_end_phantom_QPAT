from scipy.ndimage.morphology import distance_transform_edt
from utils.visualise import subfig_regression_line
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

BASE_PATH = r"..\experimental/"
COLOURS = [
    "#228833ff",  # GREEN
    "#ee6677ff",  # RED
    "#4477aaff",  # BLUE
    "#ccbb44ff"   # YELLOW
]

# can be calculated manually from utils.regression
slope = 1484.953830163365
intercept = 313.20996567364347

path = fr"{BASE_PATH}\fold_0\data.npz"
estimated_data = np.load(path)
signal = estimated_data["gt_inputs"]
gt_mua = estimated_data["gt_muas"]
gt_segmentation = estimated_data["gt_segmentations"]

bg_signals = []
bg_absorptions = []
inc_signals = []
inc_absorptions = []

for i in range(len(signal)):
    sig = np.squeeze(signal[i])
    mua = np.squeeze(gt_mua[i])
    seg = np.squeeze(gt_segmentation[i])
    bg_seg = (seg == 1)
    bg_signals.append(np.percentile(sig[bg_seg], 98))
    bg_absorptions.append(np.median(mua[bg_seg]))
    for idx in np.unique(seg):
        if idx > 1:
            target_seg = (seg == idx)
            inc_signals.append(np.percentile(sig[target_seg], 98))
            inc_absorptions.append(np.median(mua[target_seg]))

bg_absorptions = np.asarray(bg_absorptions)
bg_signals = np.asarray(bg_signals)
inc_signals = np.asarray(inc_signals)
inc_absorptions = np.asarray(inc_absorptions)

SPACING = 0.10666667
DATA_PATH = r"C:\final_data_simulation\training/"

wavelengths = np.linspace(700, 900, 21).astype(int)
phantoms = np.linspace(1, 25, 25).astype(int)

files = []

for wl in wavelengths:
    for p in phantoms:
        file = DATA_PATH + f"P.3.{p}_{wl}.npz"
        if os.path.exists(file):
            files.append(file)

depth_levels = np.linspace(5, 125, 25)
depth_signals = [None] * len(depth_levels)
depth_sim = [None] * len(depth_levels)
depth_muas = [None] * len(depth_levels)

for file in files:
    print(file)
    data = np.load(file)
    signals = data['features_das']
    segmentation = data['segmentation']
    mua = data["mua"]
    sim_sig = data['features_sim']
    distance = distance_transform_edt(segmentation)
    for d_idx, depth_level in enumerate(depth_levels):
        if depth_signals[d_idx] is None:
            depth_signals[d_idx] = []
        if depth_muas[d_idx] is None:
            depth_muas[d_idx] = []
        if depth_sim[d_idx] is None:
            depth_sim[d_idx] = []
        depth_signals[d_idx].append(np.mean(signals[(distance > depth_level-5) & (distance <= depth_level)]))
        depth_muas[d_idx].append(np.mean(mua[(distance > depth_level - 5) & (distance <= depth_level)]))
        depth_sim[d_idx].append(np.mean(sim_sig[(distance > depth_level - 5) & (distance <= depth_level)]))

depth_signals = np.asarray(depth_signals)
depth_muas = np.asarray(depth_muas)
depth_sim = np.asarray(depth_sim)

correlations = [None] * len(depth_levels)
correlations_sim = [None] * len(depth_levels)
for idx in range(len(depth_levels)):
    _, _, r_value, _, _ = linregress(depth_muas[idx], depth_signals[idx])
    correlations[idx] = r_value
    _, _, r_value_sim, _, _ = linregress(depth_muas[idx], depth_sim[idx])
    correlations_sim[idx] = r_value_sim

depth_signals = (depth_signals - np.min(depth_signals)) / (
            np.max(depth_signals) - np.min(depth_signals))
depth_sim = (depth_sim - np.min(depth_sim)) / (
            np.max(depth_sim) - np.min(depth_sim))

depth_signals_mean = np.mean(depth_signals, axis=1)
depth_signals_std = np.std(depth_signals, axis=1)
depth_sim_mean = np.mean(depth_sim, axis=1)
depth_sim_std = np.std(depth_sim, axis=1)

depth_signals_mean = (depth_signals_mean - min(depth_signals_mean)) / (max(depth_signals_mean) - min(depth_signals_mean))
depth_signals_std = (depth_signals_std - min(depth_signals_mean)) / (max(depth_signals_mean) - min(depth_signals_mean))
depth_sim_mean = (depth_sim_mean - min(depth_sim_mean)) / (max(depth_sim_mean) - min(depth_sim_mean))
depth_sim_std = (depth_sim_std - min(depth_sim_mean)) / (max(depth_sim_mean) - min(depth_sim_mean))


def plot_scatter(ax, index, yaxis=False):
    ax.set_title(f"Depth {depth_levels[index] * SPACING:.2f}mm")
    signals = depth_signals[index]
    muas = depth_muas[index]
    ax.scatter(muas, signals, s=1)
    slope, intercept, r_value, _, _ = linregress(muas, signals)
    x_values = np.linspace(min(muas), max(muas), 100)
    ax.plot(x_values, intercept + x_values * slope, "--", color="black", label=f"R={r_value:.2f}")
    if yaxis:
        ax.set_ylabel("Normalised PA Signal [a.u.]", fontweight="bold")
    else:
        ax.spines.left.set_visible(False)
        ax.set_yticks([], [])
    ax.set_xlabel("Absorption coefficient [cm$^{-1}$]", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.legend(frameon=False, loc="upper left")

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(8, 12))
ggs = fig.add_gridspec(3, 4, wspace=0.9)

f1 = fig.add_subfigure(ggs[2, 0:2])
gs = f1.add_gridspec(1, 1)
gs.update(top=0.8, bottom=0, left=0, right=0.9)
a5 = f1.add_subplot(gs[0, 0])
plot_scatter(a5, 1, yaxis=True)
a5.text(-0.2, 1.01, "D", transform=a5.transAxes, size=30, weight='bold')

# f1 = fig.add_subfigure(ggs[1, 2:4])
# gs = f1.add_gridspec(1, 1)
# gs.update(top=0.8, bottom=0, left=0, right=0.9)
# a7 = f1.add_subplot(gs[0, 0])
# plot_scatter(a7, 8)
# a7.text(-0.05, 1.01, "E", transform=a7.transAxes, size=30, weight='bold')

f1 = fig.add_subfigure(ggs[2, 2:4])
gs = f1.add_gridspec(1, 1)
gs.update(top=0.8, bottom=0, left=0, right=0.9)
a6 = f1.add_subplot(gs[0, 0])
plot_scatter(a6, -2)
a6.text(-0.05, 1.01, "E", transform=a6.transAxes, size=30, weight='bold')

f1 = fig.add_subfigure(ggs[0, 0:2])
f2 = fig.add_subfigure(ggs[0, 2:4])

subfig_regression_line(f1, bg_absorptions, (bg_signals - intercept) / slope,
                       ylabel="Background estimate", title="Cal.", color=COLOURS[3], num="A", right=0.8, top=0.9,
                       bottom=0.2)
subfig_regression_line(f2, inc_absorptions, (inc_signals - intercept) / slope,
                       ylabel="Inclusion estimate", title="Cal.", color=COLOURS[3], num="B",  right=0.8, top=0.9,
                       bottom=0.2, inclusions=True, first=False)


f1 = fig.add_subfigure(ggs[1, 0:4])
gs = f1.add_gridspec(1, 1)
gs.update(top=0.9, bottom=0.1, left=0.0, right=0.9)
a4 = f1.add_subplot(gs[0, 0])
a4.plot(depth_levels * SPACING, depth_signals_mean, label="Experiments")
a4.fill_between(depth_levels * SPACING, depth_signals_mean-depth_signals_std, depth_signals_mean+depth_signals_std,
                facecolor='blue', alpha=0.4)

a4.plot(depth_levels * SPACING, depth_sim_mean, "--", color=COLOURS[0], label="Simulation")
a4.fill_between(depth_levels * SPACING, depth_sim_mean-depth_sim_std, depth_sim_mean+depth_sim_std,
                facecolor=COLOURS[0], alpha=0.4)

a4.vlines([depth_levels[idx] * SPACING for idx in [1, -2]], 0, 1.2, "black", linestyles="--")
a4.text(depth_levels[1] * SPACING + 0.1, 0.4, "D")
# a4.text(depth_levels[8] * SPACING + 0.1, 0.4, "E")
a4.text(depth_levels[-2] * SPACING + 0.1, 0.4, "E")

a4.set_ylabel("Normalised Signal [a.u.]", fontweight="bold")
a4.set_yticks(np.arange(0.0, 1.21, 0.2), np.arange(0.0, 1.21, 0.2))
a4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
a4.set_xlabel("Depth [mm]", fontweight="bold")
a4.spines.right.set_visible(False)
a4.spines.top.set_visible(False)
a4.legend(frameon=True)
a4.text(-0.1, 1.01, "C", transform=a4.transAxes, size=30, weight='bold')



#plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(r"..\figures\figure4.png", bbox_inches='tight')
plt.savefig(r"..\figures\figure4.svg", bbox_inches='tight')
plt.savefig(r"..\figures\figure4.pdf", bbox_inches='tight')
plt.close()