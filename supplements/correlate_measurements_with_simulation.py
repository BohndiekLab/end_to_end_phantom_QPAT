from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
from utils.data_loading import PalpaitineDataset
from scipy.ndimage.morphology import distance_transform_edt

TRAINING_PATH = r"C:\final_data_simulation\training/"
train_data_exp = PalpaitineDataset(data_path=f"C:/final_data_simulation/training",
                                   augment=False, use_all_data=True, experimental_data=True)
train_data_sim = PalpaitineDataset(data_path=f"C:/final_data_simulation/training",
                                   augment=False, use_all_data=True, experimental_data=False)

exp_signals = []
sim_signals = []

for ((features_sim, _, _, _, _, _), (features_exp, _, _, _, _, seg2)) in zip(train_data_sim, train_data_exp):
    features_exp = np.squeeze(np.asarray(features_exp))
    seg2 = np.squeeze(np.asarray(seg2))
    features_sim = np.squeeze(np.asarray(features_sim))
    if len(features_exp[seg2 > 1]) > 0:
        for x in np.unique(seg2[seg2 > 1]):
            distance_image = distance_transform_edt(seg2 == x)
            exp_signals.append(np.percentile(features_exp[seg2 == x], 95))
            sim_signals.append(np.percentile(features_sim[(seg2 == x)], 95))

slope, intercept, r_value, _, _ = linregress(np.asarray(exp_signals), np.asarray(sim_signals))

print(slope)
print(intercept)
print(r_value)

plt.figure()
plt.title("Correlation of simulated and experimental inclusions")
for exp, sim in zip(exp_signals, sim_signals):
    plt.scatter(exp, sim, c="black", alpha=0.2)
plt.plot(np.sort(np.asarray(exp_signals)), intercept + slope * np.sort(np.asarray(exp_signals)),
         'b', linestyle="dotted",
         label=f"correlation (R={r_value:.2f})", alpha=0.3)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.xlabel("MSOT signal [a.u.]")
plt.ylabel("Simulated signal [a.u.]")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/supplement_sim_exp_correlation.png", dpi=300)
plt.close()
