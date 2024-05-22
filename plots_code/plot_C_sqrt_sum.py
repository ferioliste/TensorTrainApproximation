import os
if os.path.basename(os.getcwd()) != "plots_code":
    os.chdir('plots_code')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.figure(figsize=(11, 4.5))

data_path_prefix = "../test_results/"
data_file_name = "C_sqrt_sum_r.csv"
cols = ["algorithm", "rank", "error"]

df = pd.read_csv(data_path_prefix + data_file_name, usecols=cols).dropna()

tt_svd_df = df[df["algorithm"] == "tt_svd"]
tt_rsvd_df = df[df["algorithm"] == "tt_rsvd"]
STTA_gauss_df = df[df["algorithm"] == "STTA_gauss"]
STTA_srht_df = df[df["algorithm"] == "STTA_srht"]
STTA_srht_hash_df = df[df["algorithm"] == "STTA_srht_hash"]

plot_ranks = df["rank"].unique()

labels = ["TT-SVD", "TT-RSVD", "STTA-gaussian", "STTA-srht", "STTA-srht-hashed"]

plt.plot(plot_ranks, tt_svd_df.groupby("rank").error.mean(), "-o", label=labels[0], ms=3)

plot_positions = np.linspace(-1,1,6)*.3         # errorbars horizontal offsets

for i, df in enumerate([tt_rsvd_df, STTA_gauss_df, STTA_srht_df, STTA_srht_hash_df]):
    error_gb = df.groupby("rank").error
    errors05 = error_gb.quantile(0.5).values
    errors08 = error_gb.quantile(0.8).values - errors05
    errors02 = errors05 - error_gb.quantile(0.2).values

    plt.errorbar(
        plot_ranks + plot_positions[1+i],
        errors05,
        yerr=np.stack([errors02, errors08]),
        label=labels[1+i],
        capsize=3,
        linestyle="",
    )

save_path_prefix = "../plots/"
save_file_name = "C_sqrt_sum_error_r.pdf"

plt.xticks(plot_ranks)
plt.ylabel("Relative error")
plt.xlabel("TT-rank")
plt.yscale("log")
plt.legend()
plt.title("Approximation of sqaure-root-sum tensor")
plt.savefig(save_path_prefix + save_file_name, transparent=True, bbox_inches="tight")
plt.show()