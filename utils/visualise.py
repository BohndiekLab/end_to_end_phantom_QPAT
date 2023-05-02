import numpy as np
from scipy.stats import iqr, linregress
from matplotlib.ticker import FormatStrFormatter


def subfig_regression_line(fig, x, y, title="", ylabel=None, color=None, num=None, left=0.0, right=1.0, top=1.0, bottom=0.0,
                           y_axis=True, inclusions=False, first=True):
    if color is None:
        color = "black"
    if num is None:
        num = "A"

    gs = fig.add_gridspec(1, 1)
    gs.update(top=top, bottom=bottom, left=left, right=right)

    a1 = fig.add_subplot(gs[0, 0])
    a1.scatter(x, y, c=color, alpha=0.6, s=75, label=title)

    rel_error = ((np.abs(y-x) / x) * 100)
    print(title, "median", np.percentile(rel_error, 50), "iqr", iqr(rel_error))
    print("abs error:", "median", np.percentile(np.abs(y-x), 50), "iqr", iqr(np.abs(y-x)))

    _slope, _intercept, _r_value, _, _ = linregress(x, y)

    correlation_label = f"R={_r_value:.2f}"
    a1.plot(np.sort(x), _intercept + _slope * np.sort(x), 'black', linestyle="dashed",
            label=correlation_label,
            alpha=0.6)
    a1.spines.right.set_visible(False)
    a1.spines.top.set_visible(False)
    ideal_label = None
    if first:
        ideal_label = "ideal result"
    a1.plot([min(x), max(x)], [min(x), max(x)], c="black", label=ideal_label)
    a1.set_xlabel("GT absorption [cm$^{-1}$]", fontweight="bold")
    a1.legend(loc="upper left", frameon=False)
    if not y_axis:
        a1.text(-0.05, 1.01, num, transform=a1.transAxes, size=30, weight='bold')
    else:
        a1.text(-0.2, 1.01, num, transform=a1.transAxes, size=30, weight='bold')

    STEP = 0.1
    Y_MAX = 0.41
    X_MAX = 0.36
    if inclusions:
        STEP = 1.0
        Y_MAX = 5.1
        X_MAX = 4

    if y_axis:
        a1.set_ylabel(f"{ylabel} [cm$^{{-1}}$]", fontweight="bold")
        a1.set_yticks(np.arange(0, Y_MAX, STEP), np.arange(0, Y_MAX, STEP))
        a1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        a1.set_yticks([], [])
        a1.spines.left.set_visible(False)
    a1.set_ylim(0, Y_MAX)

    a1.set_xticks(np.arange(STEP, X_MAX, STEP), np.arange(STEP, X_MAX, STEP))
    a1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
