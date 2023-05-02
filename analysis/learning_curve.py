import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

plt.figure(figsize=(9, 4))
plt.title("Loss functions for all folds")
plt.gca().tick_params(direction='in')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)

colours = ["#e6194B", "#ffe119", "#42d4f4", "#4363d8", "#808000"]

plt.plot([], [], c="black", label=f"train")
plt.plot([], [], c="black", linestyle="dashed", label=f"validation")

for fold in range(5):
    path = f"../experimental/fold_{fold}/losses.npz"
    losses = np.load(path)
    training_losses = losses["training_losses"]
    validation_losses = losses["validation_losses"]

    t_x = np.linspace(0, len(validation_losses), len(training_losses))
    plt.semilogy(t_x, gaussian_filter1d(training_losses[:, 0], 50), c=colours[fold], linestyle="solid", alpha=0.5,
                 label=f"fold {fold+1}")
    plt.semilogy(validation_losses[:, 0], c=colours[fold], linestyle="dashed", alpha=0.5)

plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("../experimental/learning_rate.svg")
plt.savefig("../experimental/learning_rate.png")