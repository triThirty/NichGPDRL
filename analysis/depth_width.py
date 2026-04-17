import sys
import json
from pathlib import Path
import numpy as np

import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, ranksums


plot_name = sys.argv[1]

# folder_name = sys.argv[2]

root_analysis_path = "./record"
root_data_path = "./data"
gen_interval = 1
start_gen = 0
end_gen = 99


dir_list = [
    Path("data/046cd36bd4f_single_model_0.8/Transformer_SSGP_0.8/scenario_HH"),
    Path("data/aac1a1b190_architecture_0.8/Transformer_SSGP_0.8/scenario_HH"),
]

df = pd.DataFrame(columns=["algo", "scenario", "run", "gen", "fitness", "seed"])

for dir_name in dir_list:
    scenario_name = dir_name.stem.rsplit("_", 1)[-1]
    # experiment_name = (
    #     dir_name.parts[2] if len(dir_name.parts) > 3 else dir_name.parts[1]
    # )
    if "architecture" in dir_name.parts[1]:
        experiment_name = "Multi-Layer"
    elif "single" in dir_name.parts[1]:
        experiment_name = "Single-Layer"
    files = [f for f in dir_name.rglob("*_meng_individual_*_formula_format.json")]
    for run, file in enumerate(files):
        seed = file.stem.split("_")[0]
        with open(file, "r") as f:
            fitness_data = json.load(f)
            for gen, record in enumerate(fitness_data):
                if record["fitness"] == 0:
                    print(
                        f"{experiment_name} {scenario_name} {file} {gen} fitness is 0"
                    )
                df.loc[len(df)] = [
                    experiment_name,
                    scenario_name,
                    run,
                    gen,
                    record["fitness"],
                    seed,
                ]

avg_df = (
    df.groupby(["algo", "scenario", "gen"])["fitness"]
    .agg(mean="mean", std="std")
    .reset_index()
)

avg_df.to_csv(f"{root_analysis_path}/multi_vs_single_data.csv", index=False)


avg_df = pd.read_csv(
    Path(f"{root_analysis_path}/multi_vs_single_data.csv"),
    dtype={
        "algo": str,
        "scenario": str,
        "run": int,
        "gen": int,
        "fitness": float,
    },
)

import seaborn as sns
import matplotlib.pyplot as plt


sampled_df = avg_df.groupby(["algo", "scenario"], group_keys=True).apply(
    lambda x: x.iloc[::gen_interval]
)

multi_model = sampled_df.loc["Multi-Layer"]
single_model = sampled_df.loc["Single-Layer"]


# x_mm = multi_model.index.get_level_values(1)
# y_mm = multi_model["mean"]

# x_sm = single_model.index.get_level_values(1)
# y_sm = single_model["mean"]

# # 计算 AUC
# auc_mm = np.trapz(y_mm, x_mm)
# auc_sm = np.trapz(y_sm, x_sm)

# print("Multi-Model AUC:", auc_mm)
# print("Single-Model AUC:", auc_sm)


g = sns.relplot(
    data=sampled_df,
    x="gen",
    y="mean",
    col="scenario",
    hue="algo",
    kind="line",
    style="algo",
    markers=False,
    linewidth=1,
    alpha=0.8,
    col_wrap=1,
    facet_kws={
        "sharey": False,
    },
)
g.set_axis_labels("Generation", "Fitness")
g.set_titles("Scenario: {col_name}")
g.legend.remove()


handles, labels = g.axes[0].get_legend_handles_labels()

# Create a new legend at the bottom
g.figure.legend(
    handles=handles,
    labels=labels,
    loc="lower center",  # <--- Position the legend at the lower center of the figure
    bbox_to_anchor=(0.5, -0.05),
    ncol=len(
        labels
    ),  # <--- Arrange horizontally (number of columns equals number of legend items)
    frameon=False,  # <--- Optional: remove the legend frame
)
g.set(xlim=(start_gen, end_gen))
plt.show()
g.figure.savefig(f"{root_analysis_path}/{plot_name}.pdf", bbox_inches="tight")
