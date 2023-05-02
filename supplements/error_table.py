import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr, mannwhitneyu, linregress
from scipy.ndimage import distance_transform_edt


class GeneralisedSignalToNoiseRatio:

    def compute_measure(self, background, inclusion):
        """
        Implemented from the paper by Kempski et al 2020::
            Kempski, Kelley M., et al.
            "Application of the generalized contrast-to-noise ratio to assess photoacoustic image quality."
            Biomedical Optics Express 11.7 (2020): 3684-3698.
        This implementation uses the histogram-based approximation.
        Parameters
        ----------
        reconstructed_image
        signal_roi
            must be in the same shape as the reconstructed image
        noise_roi
            must be in the same shape as the reconstructed image
        Returns
        -------
        float, a measure of the relative overlap of the signal probability densities.
        """

        # copy input vectors as to not overwrite their contents
        background = np.copy(background)
        inclusion = np.copy(inclusion)

        # rescale signal into value range of 0 - 256 to mimic the original paper bin sizes
        signal_min = np.nanmin(background)
        signal_max = np.nanmax(inclusion)
        background = (background - signal_min) / (signal_max - signal_min)
        background = background * 256
        inclusion = (inclusion - signal_min) / (signal_max - signal_min)
        inclusion = inclusion * 256

        # define 256 unit size bins (257 bin edges) and compute the histogram PDFs.
        value_range_bin_edges = np.arange(0, 257)
        bg_hist = np.histogram(background, bins=value_range_bin_edges,
                                   density=True)[0]
        inc_hist = np.histogram(inclusion, bins=value_range_bin_edges,
                                  density=True)[0]

        # compute the overlap
        overlap = 0
        for i in range(256):
            overlap = overlap + np.min([bg_hist[i], inc_hist[i]])

        # return gCNR
        return 1 - overlap

    def get_name(self):
        return "gCNR"


GCNR = GeneralisedSignalToNoiseRatio()
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

bg_absorptions = []
bg_fluence_corrected = []
est_sim_bg_absorptions = []
est_exp_bg_absorptions = []

inc_absorptions = []
inc_fluence_corrected = []
est_sim_inc_absorptions = []
est_exp_inc_absorptions = []

gCNR_sig = []
gCNR_phi = []
gCNR_exp = []
gCNR_sim = []

wavelengths_inc = []
wavelengths_bg = []

in_distribution = [np.zeros((21, )), np.ones((42, )), np.zeros((21, )), np.ones((6*21, )), np.zeros((21, )), np.ones((21, )), np.zeros((8*21, ))]
in_distribution = (np.hstack(in_distribution)).astype(bool)


wavelengths = np.arange(700, 901, 10)
for i in range(len(signal)):
    wl = wavelengths[i%len(wavelengths)]
    # extract the values for this image
    sig = (np.squeeze(signal[i]))
    # if i%len(wavelengths) == 0:
    #     plt.imshow(sig)
    #     plt.show()
    phi_sig = np.abs(sig) / fluences[i]
    est_mua_exp_img = np.squeeze(est_mua_exp[i])
    est_mua_sim_img = np.squeeze(est_mua_sim[i])
    mua = np.squeeze(gt_mua[i])
    seg = np.squeeze(gt_segmentation[i])

    # save all the background aggregates
    bg_seg = (seg == 1)
    bg_fluence_corrected.append(np.median(phi_sig[bg_seg]))
    bg_absorptions.append(np.median(mua[bg_seg]))
    est_exp_bg_absorptions.append(np.median(est_mua_exp_img[bg_seg]))
    est_sim_bg_absorptions.append(np.median(est_mua_sim_img[bg_seg]))

    phi_bg_distribution = phi_sig[bg_seg]
    exp_bg_distribution = est_mua_exp_img[bg_seg]
    sim_bg_distribution = est_mua_sim_img[bg_seg]
    sig_bg_distribution = sig[bg_seg]

    wavelengths_bg.append(wl)

    # save all inclusion aggregated
    for idx in np.unique(seg):
        if idx > 1:
            target_seg = (seg == idx)
            distances = distance_transform_edt(target_seg)
            inc_fluence_corrected.append(np.mean(phi_sig[target_seg & (distances < DISTANCE_THRESHOLD)]))
            inc_absorptions.append(np.median(mua[target_seg]))
            est_exp_inc_absorptions.append(np.mean(est_mua_exp_img[target_seg & (distances < DISTANCE_THRESHOLD)]))
            est_sim_inc_absorptions.append(np.mean(est_mua_sim_img[target_seg & (distances < DISTANCE_THRESHOLD)]))
            wavelengths_inc.append(wl)

            sig_inc_dist = sig[target_seg & (distances < DISTANCE_THRESHOLD)]
            phi_inc_dist = phi_sig[target_seg & (distances < DISTANCE_THRESHOLD)]
            exp_inc_dist = est_mua_exp_img[target_seg & (distances < DISTANCE_THRESHOLD)]
            sim_inc_dist = est_mua_sim_img[target_seg & (distances < DISTANCE_THRESHOLD)]

            gCNR_sig.append(GCNR.compute_measure(sig_bg_distribution, sig_inc_dist))
            gCNR_phi.append(GCNR.compute_measure(phi_bg_distribution, phi_inc_dist))
            gCNR_exp.append(GCNR.compute_measure(exp_bg_distribution, exp_inc_dist))
            gCNR_sim.append(GCNR.compute_measure(sim_bg_distribution, sim_inc_dist))

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

gCNR_sig = np.asarray(gCNR_sig)
gCNR_phi = np.asarray(gCNR_phi)
gCNR_exp = np.asarray(gCNR_exp)
gCNR_sim = np.asarray(gCNR_sim)

bg_errors_phi = np.abs(bg_fluence_corrected - bg_absorptions)
bg_errors_exp = np.abs(est_exp_bg_absorptions - bg_absorptions)
bg_errors_sim = np.abs(est_sim_bg_absorptions - bg_absorptions)

inc_errors_phi = np.abs(inc_fluence_corrected - inc_absorptions)
inc_errors_exp = np.abs(est_exp_inc_absorptions - inc_absorptions)
inc_errors_sim = np.abs(est_sim_inc_absorptions - inc_absorptions)

bg_errors_phi_rel = (bg_errors_phi / bg_absorptions) * 100
bg_errors_exp_rel = (bg_errors_exp / bg_absorptions) * 100
bg_errors_sim_rel = (bg_errors_sim / bg_absorptions) * 100

inc_errors_phi_rel = (inc_errors_phi / inc_absorptions) * 100
inc_errors_exp_rel = (inc_errors_exp / inc_absorptions) * 100
inc_errors_sim_rel = (inc_errors_sim / inc_absorptions) * 100

_, _, bg_r_phi, _, _ = linregress(bg_fluence_corrected, bg_absorptions)
_, _, bg_r_exp, _, _ = linregress(est_exp_bg_absorptions, bg_absorptions)
_, _, bg_r_sim, _, _ = linregress(est_sim_bg_absorptions, bg_absorptions)

_, _, inc_r_phi, _, _ = linregress(inc_fluence_corrected, inc_absorptions)
_, _, inc_r_exp, _, _ = linregress(est_exp_inc_absorptions, inc_absorptions)
_, _, inc_r_sim, _, _ = linregress(est_sim_inc_absorptions, inc_absorptions)

_, _, inc_r_phi_low, _, _ = linregress(inc_fluence_corrected[inc_absorptions < 2.5], inc_absorptions[inc_absorptions < 2.5])
_, _, inc_r_exp_low, _, _ = linregress(est_exp_inc_absorptions[inc_absorptions < 2.5], inc_absorptions[inc_absorptions < 2.5])
_, _, inc_r_sim_low, _, _ = linregress(est_sim_inc_absorptions[inc_absorptions < 2.5], inc_absorptions[inc_absorptions < 2.5])
inc_errors_phi_low = np.abs(inc_fluence_corrected[inc_absorptions < 2.5] - inc_absorptions[inc_absorptions < 2.5])
inc_errors_exp_low = np.abs(est_exp_inc_absorptions[inc_absorptions < 2.5] - inc_absorptions[inc_absorptions < 2.5])
inc_errors_sim_low = np.abs(est_sim_inc_absorptions[inc_absorptions < 2.5] - inc_absorptions[inc_absorptions < 2.5])
inc_errors_phi_rel_low = (inc_errors_phi[inc_absorptions < 2.5] / inc_absorptions[inc_absorptions < 2.5]) * 100
inc_errors_exp_rel_low = (inc_errors_exp[inc_absorptions < 2.5] / inc_absorptions[inc_absorptions < 2.5]) * 100
inc_errors_sim_rel_low = (inc_errors_sim[inc_absorptions < 2.5] / inc_absorptions[inc_absorptions < 2.5]) * 100
gCNR_phi_low = gCNR_phi[inc_absorptions < 2.5]
gCNR_exp_low = gCNR_exp[inc_absorptions < 2.5]
gCNR_sim_low = gCNR_sim[inc_absorptions < 2.5]

table = (rf"\begin{{table}}[h!tb]" "\n"
         rf"\centering" "\n"
         rf"\caption{{Tabulated performance results of the different methods. Values are shown as median $\pm$ interquartile range/2. R = Pearson correlation coefficient}}" "\n"
         rf"\label{{tab:my_label}}" "\n"
         rf"\begin{{tabular}}{{lcccc}}" "\n"
         rf"\textbf{{Method}} & R & gCNR & Rel. Err. [\%] & Abs. Err. [cm$^{{-1}}$] \\" "\n"
         rf"\hline" "\n"
         rf"\\" "\n"
         rf"\textbf{{Background}} \\" "\n"
         rf"\\" "\n"
         rf"Cal. Signal & 0.88 & N/A & 25 $\pm$ 17 & 0.03 $\pm$ 0.02\\" "\n"
         rf"GT-$\phi$ & {bg_r_phi:.2f} & N/A & {np.median(bg_errors_phi_rel):.0f} $\pm$ {iqr(bg_errors_phi_rel)/2:.0f} & {np.median(bg_errors_phi):.2f} $\pm$ {iqr(bg_errors_phi)/2:.2f}\\" "\n"
         rf"DL-Sim & {bg_r_sim:.2f} & N/A & {np.median(bg_errors_sim_rel):.0f} $\pm$ {iqr(bg_errors_sim_rel)/2:.0f} & {np.median(bg_errors_sim):.2f} $\pm$ {iqr(bg_errors_sim)/2:.2f}\\" "\n"
         rf"DL-Exp & {bg_r_exp:.2f} & N/A & {np.median(bg_errors_exp_rel):.0f} $\pm$ {iqr(bg_errors_exp_rel)/2:.0f} & {np.median(bg_errors_exp):.2f} $\pm$ {iqr(bg_errors_exp)/2:.2f}\\" "\n"
         rf"\\" "\n"
         rf"\textbf{{All Inclusions}} \\" "\n"
         rf"\\" "\n"
         rf"Cal. Signal & -0.22 & {np.median(gCNR_sig):.2f} $\pm$ {iqr(gCNR_sig)/2:.1f}& 83 $\pm$ 30 & 1.15 $\pm$ 1.0\\" "\n"
         rf"GT-$\phi$ & {inc_r_phi:.2f} & {np.median(gCNR_phi):.2f} $\pm$ {iqr(gCNR_phi)/2:.1f} & {np.median(inc_errors_phi_rel):.0f} $\pm$ {iqr(inc_errors_phi_rel)/2:.0f} & {np.median(inc_errors_phi):.2f} $\pm$ {iqr(inc_errors_phi)/2:.1f}\\" "\n"
         rf"DL-Sim & {inc_r_sim:.2f} & {np.median(gCNR_sim):.2f} $\pm$ {iqr(gCNR_sim)/2:.1f} & {np.median(inc_errors_sim_rel):.0f} $\pm$ {iqr(inc_errors_sim_rel)/2:.0f} & {np.median(inc_errors_sim):.2f} $\pm$ {iqr(inc_errors_sim)/2:.1f}\\" "\n"
         rf"DL-Exp & {inc_r_exp:.2f} & {np.median(gCNR_exp):.2f} $\pm$ {iqr(gCNR_exp)/2:.1f} & {np.median(inc_errors_exp_rel):.0f} $\pm$ {iqr(inc_errors_exp_rel)/2:.0f} & {np.median(inc_errors_exp):.2f} $\pm$ {iqr(inc_errors_exp)/2:.1f}\\" "\n"
         rf"\\" "\n"
         rf"\textbf{{Inc. $\mu_a\leq2.5$}} \\" "\n"
         rf"\\" "\n"
         rf"GT-$\phi$ & {inc_r_phi_low:.2f} & {np.median(gCNR_phi_low):.2f} $\pm$ {iqr(gCNR_phi_low)/2:.1f} & {np.median(inc_errors_phi_rel_low):.0f} $\pm$ {iqr(inc_errors_phi_rel_low)/2:.0f} & {np.median(inc_errors_phi_low):.2f} $\pm$ {iqr(inc_errors_phi_low)/2:.1f}\\" "\n"
         rf"DL-Sim & {inc_r_sim_low:.2f} & {np.median(gCNR_sim_low):.2f} $\pm$ {iqr(gCNR_sim_low)/2:.1f} & {np.median(inc_errors_sim_rel_low):.0f} $\pm$ {iqr(inc_errors_sim_rel_low)/2:.0f} & {np.median(inc_errors_sim_low):.2f} $\pm$ {iqr(inc_errors_sim_low)/2:.1f}\\" "\n"
         rf"DL-Exp & {inc_r_exp_low:.2f} & {np.median(gCNR_exp_low):.2f} $\pm$ {iqr(gCNR_exp_low)/2:.1f} & {np.median(inc_errors_exp_rel_low):.0f} $\pm$ {iqr(inc_errors_exp_rel_low)/2:.0f} & {np.median(inc_errors_exp_low):.2f} $\pm$ {iqr(inc_errors_exp_low)/2:.1f}\\" "\n"
         rf"\\" "\n"
         rf"\end{{tabular}}" "\n"
         rf"\end{{table}}" "\n")

print(table)