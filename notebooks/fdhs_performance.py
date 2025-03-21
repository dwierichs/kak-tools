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

# Config
#workflow_level = "complete"
#workflow_level = "known_algebra"
workflow_level = "minimal"

coefficients = "random"
spacing = 0.3
if workflow_level == "complete":
    a = np.log(2)
    b = np.log(100)
    base_num_reps = 300
    workflow = complete_workflow_tfXY
    start_fit = 6
elif workflow_level == "known_algebra":
    a = np.log(2)
    b = np.log(200)
    base_num_reps = 300
    workflow = workflow_tfXY_known_algebra
    start_fit = 7
elif workflow_level == "minimal":
    a = np.log(2)
    b = np.log(1000)
    base_num_reps = 3000
    workflow = minimal_workflow_tfXY
    start_fit = 3

ns = np.round(np.exp(np.linspace(a, b, int((b - a) / spacing))))
num_reps = {n: int(np.ceil(base_num_reps/ (n / 4)**2)) for n in ns}
print(ns)

n_max = int(ns[-1])
t0 = 0.1
recompute = True
save = True
data_filename = f"/home/david/repos/kak-tools/data/fdhs_performance_{workflow_level}_{n_max}"
plot_filename = f"/home/david/repos/kak-tools/gfx/fdhs_performance_{workflow_level}.pdf"

if recompute:
    Start = time.process_time()
    Times = []
    if save:
        np.save(data_filename+"_ns.npy", ns)

    for n in ns:
        n = int(n)
        n_so = 2 * n
        so_dim = (n_so**2-n_so) // 2

        times = []
        for _ in tqdm(range(num_reps[n])):
            start = time.process_time()
            pauli_decomp = workflow(n, t0, coefficients)
            end = time.process_time()
            times.append(end - start)

        Times.append(np.mean(times))
        print(f"Decomposed exp(iHt) on {n} qubits into {len(pauli_decomp)} Pauli rotations (DLA dimension: {so_dim}, Duration: {Times[-1]:.3}s).")
    all_times = np.array(Times)
    End = time.process_time()
    print(f"Overall took {(End-Start) / 60} minutes.")
    if save:
        np.save(data_filename+".npy", all_times)
else:
    all_times = np.load(data_filename+".npy")

fig, axs = plt.subplots(1, 1, figsize=(7, 5))

f = lambda x, a, b: (b * x)**a
f_log = lambda x, a, b: a * x + a * np.log(b)
skip_fit = False

try:
    #popt, pcov = curve_fit(f, ns[:len(all_times)], all_times)
    popt, pcov = curve_fit(f_log, np.log(ns[start_fit:len(all_times)]), np.log(all_times[start_fit:]))
except:
    skip_fit = True
cont_ns = np.linspace(ns[0], ns[-1], 100)

ax = axs
c = "xkcd:blue violet"
ax.plot(ns[:len(all_times)], all_times, marker="s", c=c, label="Data", ls="")
# ax.plot(cont_ns, np.exp(f(np.log(cont_ns), *popt)), ls = "--", c="xkcd:red pink", label=f"${popt[1]:.1f}n^{{{popt[0]:.2f}}}$")
if not skip_fit:
    ax.plot(cont_ns, f(cont_ns, *popt), ls = "--", c=c, label=f"Fit: $({popt[1]:.3f}n)^{{{popt[0]:.2f}}}$")
ax.set_xlabel("Number of qubits $n$")
ax.set_ylabel("Decomposition runtime $t$ / s")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

plt.savefig(plot_filename, dpi=300)
plt.show()
