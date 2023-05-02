import glob
import json
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("error_propagation/sims/*_low.npz")
with open("error_propagation/data/thickness.json", "r+") as json_file:
    thicknesses = json.load(json_file)

def calc_thickness_std_percent(name):
    a = name + ".a/"
    b = name + ".a/"

    def load_p(n):
        results = []
        if n in thicknesses:
            t_dict = thicknesses[n]
            keys = t_dict.keys()
            for key in keys:
                mean = np.mean(t_dict[key])
                std = np.std(t_dict[key])
                results.append((std/mean) * 100)
        else:
            return []
        return results

    percentages_a = load_p(a)
    percentages_b = load_p(b)

    return np.mean(np.stack([percentages_a, percentages_b]))

t_diffs = []
mua_diffs = []
p0_diffs = []
recon_diffs = []

for sim_result in files:
    print(sim_result)
    name = sim_result.split("/")[-1].split("\\")[-1].split("_")[0]
    print(name)
    t_diffs.append(calc_thickness_std_percent(name))

    def load(data):
        return (data["mua"].item(),
                data["mus"].item(),
                data["p0"],
                data["recon"])

    mua_low, mus_low, p0_low, recon_low = load(np.load(f"error_propagation/sims/{name}_low.npz"))
    mua_mean, mus_mean, p0_mean, recon_mean = load(np.load(f"error_propagation/sims/{name}_mean.npz"))
    mua_high, mus_high, p0_high, recon_high = load(np.load(f"error_propagation/sims/{name}_high.npz"))

    actual_p0 = ((np.abs(p0_high - p0_low) / 2) / p0_mean) * 100
    actual_recon = ((np.abs(recon_high - recon_low) / 2) / recon_mean) * 100

    mua_diffs.append((((mua_high - mua_low) / 2) / mua_mean) * 100)
    p0_diffs.append(np.percentile(actual_p0, 67))  # circa 33% of the pixels are outside of the volume.
    recon_diffs.append(np.percentile(actual_recon, 67))  # so this represents roughly the median over the remaining values

rnd = np.random.uniform(size=len(t_diffs))*0.6-0.3
plt.figure(figsize=(8, 4))

for i in range(len(t_diffs)):
    plt.plot(np.asarray([1, 2, 3, 4]) + rnd[i],
             [t_diffs[i], mua_diffs[i], p0_diffs[i], recon_diffs[i]],
             color="black", alpha=0.4)
plt.scatter(np.ones(len(t_diffs)) + rnd, t_diffs)
plt.boxplot(t_diffs, positions=[1], widths=[0.6], showfliers=False, labels=["Thickness\nmeasurement"])
plt.scatter(np.ones(len(mua_diffs)) * 2 + rnd, mua_diffs)
plt.boxplot(mua_diffs, positions=[2], widths=[0.6], showfliers=False, labels=["Absorption\ncoefficient"])
plt.scatter(np.ones(len(p0_diffs)) * 3 + rnd, p0_diffs)
plt.boxplot(p0_diffs, positions=[3], widths=[0.6], showfliers=False, labels=["Initial\npressure"])
plt.scatter(np.ones(len(recon_diffs)) * 4 + rnd, recon_diffs)
plt.boxplot(recon_diffs, positions=[4], widths=[0.6], showfliers=False, labels=["Reconstructed\nsignals"])
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.ylabel("Standard Error [%]")
plt.tight_layout()
plt.savefig("suppl_figure8.png", dpi=300)
