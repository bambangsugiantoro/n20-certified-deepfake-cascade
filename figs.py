"""Figures for the Cluster Computing manuscript. All numbers are copied from the HPC result files
(results/face_cc2/summary.csv, face_family2/per_family.csv, system/hpc_bench.json [job 1655], edge/edge_results.json)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .3})
C = {"C": "#1f4e79", "SETW": "#2e86c1", "A": "#c0392b", "B": "#e67e22", "H": "#7f8c8d"}
n = np.array([800, 3200, 6400, 12800])
# face_cc2, orig condition, class-conditional risk (alpha = 0.05)
rel = {0.5: dict(C=[.3503, .6117, .6845, .7288], SETW=[.3405, .6094, .6827, .7261], A=[.1083, .3492, .6000, .6577], B=[.1083, .4997, .6329, .6758]),
       0.1: dict(C=[.0883, .3765, .4835, .5355], SETW=[.0745, .3715, .4790, .5323], A=[.0403, .1903, .2605, .3005], B=[.0403, .1878, .2498, .2838])}
cost = {0.5: dict(A=[145.5, 121.0, 117.7, 112.4], B=[141.8, 123.1, 122.6, 116.4]), 0.1: dict(A=[156.6, 132.2, 120.9, 114.5], B=[152.6, 129.2, 119.4, 114.0])}
lab = {"C": "C (tier 3 only)", "SETW": "Certified set (SETW)", "A": "A (1→2→3)", "B": "B (1→3)"}

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for a, p in zip(ax, (0.5, 0.1)):
    for k in ("C", "SETW", "B", "A"):
        a.plot(n, np.array(rel[p][k]) * 100, marker="o", ms=3.5, lw=1.4, color=C[k], ls="--" if k == "SETW" else "-", label=lab[k])
    a.set_xscale("log", base=2); a.set_xticks(n); a.set_xticklabels([f"{v:,}" for v in n])
    a.set_title(f"Prevalence of fakes $\\pi$ = {int(p*100)}%"); a.set_xlabel("Target labels $n$")
ax[0].set_ylabel("Certified release (%)"); ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig("Fig2.pdf"); plt.close(fig)

# compute-coverage trade-off (orig, pi = 0.5): each point = one label budget
fig, a = plt.subplots(figsize=(3.4, 2.7))
for k, mk in (("A", "o"), ("B", "s")):
    a.plot(cost[0.5][k], np.array(rel[0.5][k]) * 100, marker=mk, color=C[k], lw=1.2, label=lab[k])
    for x, y, m in zip(cost[0.5][k], np.array(rel[0.5][k]) * 100, n):
        if m in (3200, 12800): a.annotate(f"n={m:,}", (x, y), textcoords="offset points", xytext=(-30, -11) if k == "A" else (4, 3), fontsize=6.5, color=C[k])
a.scatter([158.8] * 4, np.array(rel[0.5]["C"]) * 100, color=C["C"], marker="D", s=18, label=lab["C"], zorder=3)
a.set_xlabel("Mean compute per image (GFLOPs)"); a.set_ylabel("Certified release (%)")
a.set_xlim(105, 165); a.legend(frameon=False, fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig("Fig3.pdf"); plt.close(fig)

# per-family adapted error (orig, pi = 0.5, n = 12,800; face_family2)
fam = ["DF40 reenactment", "DF40 diffusion/AR", "DF40 face edit", "DiFF I2I", "DiFF face edit", "DiFF T2I", "DiFF face swap",
       "Real FF++/CDF", "Real DiFF", "Real DF40"]
t1 = [30.4, 24.2, 26.1, 20.6, 24.8, 22.1, 30.6, 29.8, 18.1, 22.6]
t3 = [18.8, 11.6, 6.5, 15.1, 10.1, 7.9, 5.6, 23.3, 10.5, 6.7]
rc = [70.3, 80.9, 89.0, 73.2, 83.4, 84.8, 89.2, 48.1, 74.9, 82.7]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True); y = np.arange(len(fam))[::-1]
ax[0].barh(y + .18, t1, .36, color=C["A"], label="Tier 1 (MobileNetV3-L)"); ax[0].barh(y - .18, t3, .36, color=C["C"], label="Tier 3 (CLIP+DINOv2)")
ax[0].set_yticks(y); ax[0].set_yticklabels(fam); ax[0].set_xlabel("Adapted error (%)"); ax[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=2)
ax[1].barh(y, rc, .55, color=C["SETW"]); ax[1].set_xlabel("Certified release, configuration C (%)"); ax[1].set_xlim(0, 100)
for yy, v in zip(y, rc): ax[1].text(v + 1, yy, f"{v:.0f}", va="center", fontsize=6.5)
fig.tight_layout(); fig.savefig("Fig4.pdf"); plt.close(fig)

# edge vs cluster: latency (batch 1, p50) and GPU energy per image (batch 8)
lbl = ["T1\nedge GPU", "T2\nedge GPU", "T1\nedge CPU", "T1\nL40S", "T2\nL40S", "T3\nL40S"]
lat = [6.58, 5.73, 14.59, 3.0, 2.6, 10.9]
en = [15.6, 37.8, np.nan, 1.5, 7.1, 466.5]
col = ["#c0392b", "#e67e22", "#d98880", "#1f4e79", "#2e86c1", "#154360"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax[0].bar(lbl, lat, color=col); ax[0].set_ylabel("Latency, batch 1, p50 (ms)")
for a_ in ax: a_.tick_params(axis="x", labelsize=7)
ax[0].axhline(39.2, color="k", ls=":", lw=1); ax[0].text(5.4, 40.5, "edge→cluster RTT 39 ms", ha="right", fontsize=7)
ax[0].set_ylim(0, 48)
ax[1].bar([l for l, e in zip(lbl, en) if not np.isnan(e)], [e for e in en if not np.isnan(e)], color=[c for c, e in zip(col, en) if not np.isnan(e)])
ax[1].set_yscale("log"); ax[1].set_ylabel("GPU energy/image, batch 8 (mJ)"); ax[1].set_ylim(1, 1500)
for i, e in enumerate([e for e in en if not np.isnan(e)]): ax[1].text(i, e * 1.15, f"{e:g}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig("Fig5.pdf"); plt.close(fig)
print("ok")
