import os
if os.path.basename(os.getcwd()) != "plots_code":
    os.chdir('plots_code')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_path_prefix = "../test_results/"
data_file_name = "A_hilbert_r.csv"
cols = ["rank", "algorithm", "t1", "t2", "t3", "time_taken", "error"]

df = pd.read_csv(data_path_prefix + data_file_name, usecols=cols).dropna()

file_names = ["A_hilbert_tt_svd_ratio_r", "A_hilbert_tt_rsvd_ratio_r"]
labels = [["svd", "matrix multiplication", None, "other"], ["sketching", "QR factorization", "matrix multiplication", "other"]]

colors = ["#fad132", "#ff5500", "#ff96f1", "#dc0000"]
algorithms = ["tt_svd", "tt_rsvd"]

rank = None
time_taken = []
errors05 = []
errors08 = []
errors02 = []
for alg_id, alg in enumerate(algorithms):  
    alg_df = df[df['algorithm'] == alg].sort_values(by='rank').drop(columns=['algorithm']).groupby("rank")
    rank  = list(alg_df.groups.keys())
    
    t1 = alg_df.t1.quantile(0.5).values
    t2 = alg_df.t2.quantile(0.5).values
    t3 = alg_df.t3.quantile(0.5).values
    time_taken.append(alg_df.time_taken.quantile(0.5).values)
    
    errors05.append(alg_df.error.quantile(0.5).values)
    errors02.append(errors05[alg_id] - alg_df.error.quantile(0.2).values)
    errors08.append(alg_df.error.quantile(0.8).values - errors05[alg_id])
    
    percentages = [[t1[i]/time_taken[alg_id][i], t2[i]/time_taken[alg_id][i], t3[i]/time_taken[alg_id][i], (time_taken[alg_id][i]-t1[i]-t2[i]-t3[i])/time_taken[alg_id][i]] for i in range(len(rank))]
    
    fig, ax = plt.subplots()
    for i, row in enumerate(percentages):
        for j in range(len(row)):
            if j == 0:
                ax.bar(i, row[j], label = (labels[alg_id][j] if i == 0 else None), width=1.05, color=colors[j])
            else:
                bottom = sum(row[:j])
                ax.bar(i, row[j], bottom=bottom, label = (labels[alg_id][j] if i == 0 else None), width=1.05, color=colors[j])
    
    ax.set_ylabel('Runtime ripartition')
    ax.set_xlabel('$TT-rank$')
    ax.set_title('Factorization of Hilbert tensor ($d=5$, $n=5$) with ' + alg)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xticklabels([1, 5, 9, 13, 17])
    ax.legend(loc='lower left')

    save_path = "../plots/"
    plt.savefig(save_path + file_names[alg_id] + ".pdf", transparent=True, bbox_inches="tight")
    plt.close('all')





labels = ["tt_svd", "tt_rsvd"]
file_name = "A_hilbert_time_r"

plt.plot(rank, time_taken[0], "-", label=labels[0], ms=3)
plt.plot(rank, time_taken[1], "-", label=labels[1], ms=3)

plt.xticks(rank)
plt.ylabel("Runtime")
plt.xlabel("TT-rank")
plt.legend()
plt.title("Factorization of Hilbert tensor ($d=5$, $n=5$)")

save_path = "../plots/"
plt.savefig(save_path + file_name + ".pdf", transparent=True, bbox_inches="tight")
plt.close('all')







labels = ["tt_svd", "tt_rsvd"]
file_name = "A_hilbert_error_r"

plt.plot(rank, errors05[0], "-o", label=labels[0], ms=3)

plt.errorbar(
        rank,
        errors05[1],
        yerr=np.stack([errors02[1], errors08[1]]),
        label=labels[1],
        capsize=3,
        linestyle="",
    )

plt.xticks(rank)
plt.ylabel("Relative error")
plt.xlabel("TT-rank")
plt.yscale("log")
plt.legend()
plt.title("Factorization of Hilbert tensor ($d=5$, $n=5$)")

save_path = "../plots/"
plt.savefig(save_path + file_name + ".pdf", transparent=True, bbox_inches="tight")
plt.close('all')