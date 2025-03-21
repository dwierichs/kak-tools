from itertools import combinations, product
import numpy as np
import time
np.random.seed(2415)
from scipy.linalg import expm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
jax.config.update("jax_enable_x64", True)

import pennylane as qml
from pennylane import X, Y, Z, I

from kak_tools.full_workflows import minimal_workflow_tfXY, complete_workflow_tfXY, workflow_tfXY_known_algebra

#plt.rcParams["font.family"] = "serif"  
#plt.rcParams["font.family"] = "serif"  
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
plt.rcParams["text.latex.preamble"] += r"\usepackage{amssymb}\usepackage{amsmath}\usepackage{siunitx}"

# Config
plot_filename = f"/home/david/repos/kak-tools/gfx/fdhs_performance.pdf"
fontsize = 18

workflow_levels = ["complete", "known_algebra", "minimal", "diag_only"]
#Colors = {"complete": "xkcd:blue violet", "known_algebra": "xkcd:red pink", "minimal": "xkcd:orange yellow", "diag_only": "xkcd:moss"}
Colors = {"complete": "#9411a7", "known_algebra": "#f27675", "minimal": "#ffa82a", "diag_only": "#44aa00"}
Markers = {"complete": "s", "known_algebra": "o", "minimal": "d", "diag_only": "+"}
Labels = {"complete": "Complete", "known_algebra": r"Known $\mathfrak{g}$", "minimal": "Minimal", "diag_only": "Diag."}
N_max = {"complete": 100, "known_algebra": 200, "minimal": 1000, "diag_only": 2000}
End_fit = {"complete": 6, "known_algebra": 8, "minimal": 0, "diag_only": 0}
Start_fit = {"complete": 7, "known_algebra": 9, "minimal": 0, "diag_only": 10}

f = lambda x, a, b: b * x**a
f_log = lambda x, a, b: a * x + np.log(b)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for workflow_level in workflow_levels:
    n_max = N_max[workflow_level]
    ns = np.load(f"/home/david/repos/kak-tools/data/fdhs_performance_{workflow_level}_{n_max}_ns.npy")
    #ns = 2 * ns**2 - ns
    times = np.load(f"/home/david/repos/kak-tools/data/fdhs_performance_{workflow_level}_{n_max}.npy")

    m = Markers[workflow_level]
    c = Colors[workflow_level]
    data_label = Labels[workflow_level]

    ax.plot(ns, times, marker=m, c=c, label=data_label, ls="")

for workflow_level in workflow_levels:
    n_max = N_max[workflow_level]
    ns = np.load(f"/home/david/repos/kak-tools/data/fdhs_performance_{workflow_level}_{n_max}_ns.npy")
    #ns = 2 * ns**2 - ns
    times = np.load(f"/home/david/repos/kak-tools/data/fdhs_performance_{workflow_level}_{n_max}.npy")
    cont_ns = np.linspace(ns[0], ns[-1], 100)

    m = Markers[workflow_level]
    c = Colors[workflow_level]
    data_label = Labels[workflow_level]

    # Fits
    end_fit = End_fit[workflow_level]
    start_fit = Start_fit[workflow_level]
    popt, pcov = curve_fit(f_log, np.log(ns[start_fit:]), np.log(times[start_fit:]))
    ax.plot(cont_ns, f(cont_ns, *popt), ls = "--", c=c, label=f"$\\num{{{np.format_float_scientific(popt[1], precision=1)}}}\,n^{{{popt[0]:.2f}}}$")
    #if end_fit > 0:
        #popt, pcov = curve_fit(f_log, np.log(ns[:end_fit]), np.log(times[:end_fit]))
        #ax.plot(cont_ns, f(cont_ns, *popt), ls = ":", c=c, label=f"$({popt[1]:.3f}n)^{{{popt[0]:.2f}}}$")
ax.set_xlabel("Number of qubits $n$", fontsize=fontsize)
ax.set_ylabel("Decomposition runtime $t$ [s]", fontsize=fontsize)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=fontsize, ncol=2, columnspacing=0.3, handletextpad=0.2)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)
plt.show()

