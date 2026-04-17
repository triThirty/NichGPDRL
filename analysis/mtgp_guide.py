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
target_algo = "MTGP_0.8"
gen_interval = 1
start_gen = 0
end_gen = 99


dir_list = [
    Path(f"data/6d31922_dup_removal_MTGP/{target_algo}/scenario_HH"),
    Path(f"data/c0ab00be_dup_removal_score_guide_MTGP/{target_algo}/scenario_HH"),
]

df = pd.DataFrame(columns=["algo", "scenario", "run", "gen", "fitness", "seed"])

for dir_name in dir_list:
    scenario_name = dir_name.stem.rsplit("_", 1)[-1]
    # experiment_name = (
    #     dir_name.parts[2] if len(dir_name.parts) > 3 else dir_name.parts[1]
    # )
    if "score" in dir_name.parts[1]:
        experiment_name = "Score-guide MTGP"
    else:
        experiment_name = "MTGP"
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

# multi_model = sampled_df.loc["Multi-Model"]
# single_model = sampled_df.loc["Single-Model"]


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
# g.set(yscale="log")
g.set_axis_labels("Generation", "Fitness")
g.set_titles("Scenario: {col_name}")
g.legend.remove()


handles, labels = g.axes[0].get_legend_handles_labels()

# Create a new legend at the bottom
g.figure.legend(
    handles=handles,
    labels=labels,
    loc="lower center",  # <--- Position the legend at the lower center of the figure
    # bbox_to_anchor=(0.5, -0.05),
    ncol=len(
        labels
    ),  # <--- Arrange horizontally (number of columns equals number of legend items)
    frameon=False,  # <--- Optional: remove the legend frame
)
# plt.figure(figsize=(4, 3))
g.set(xlim=(start_gen, end_gen))
# for ax in g.axes.flatten():
#     scenario = ax.get_title().split(": ")[-1]
#     ax.set_ylim(
#         avg_df.query(f"scenario=='{scenario}'").min()["mean"] - 10,
#         avg_df.query(f"scenario=='{scenario}' and gen=={start_gen}").max()["mean"],
#     )
plt.show()
g.figure.savefig(f"{root_analysis_path}/{plot_name}.pdf", bbox_inches="tight")


# last_gen_df = avg_df.query(f"gen == {end_gen}").reset_index(drop=True)
# last_gen_df["Mean (Std)"] = last_gen_df.apply(
#     lambda row: f"{row['mean']:.2f}({row['std']:.2f})", axis=1
# )
# print("\nDataFrame after filter (last_gen_df):\n", last_gen_df)

# pivot_table = last_gen_df.pivot_table(
#     index="algo",
#     columns="scenario",
#     values="Mean (Std)",
#     aggfunc="first",  # Use 'first' since each scenario-algo pair has only one value
# )

# pivot_table["Rank"] = pivot_table.rank().mean(axis=1, numeric_only=False)

# # Optional: Rename the index 'scenario' to 'Scenarios' for clarity

# unique_scenarios = avg_df["scenario"].unique()
# unique_algos = avg_df["algo"].unique()

# print(f"Unique Scenarios: {unique_scenarios.tolist()}")
# print(f"Unique Algorithms: {unique_algos.tolist()}\n")

# # Store results
# wilcoxon_results = []
# alpha = 0.05  # Significance level

# # Iterate through each scenario
# for scenario in unique_scenarios:
#     print(f"--- Scenario: {scenario} ---")
#     scenario_df = df[df["scenario"] == scenario]

#     # Iterate through each pair of algorithms
#     for algo1 in unique_algos[unique_algos != target_algo]:
#         data_algo1 = scenario_df.query(f"gen=={end_gen} and algo==@algo1")["fitness"]
#         data_algo2 = scenario_df.query(f"gen=={end_gen} and algo=='{target_algo}'")[
#             "fitness"
#         ]

#         if len(data_algo1) == 0 or len(data_algo2) == 0:
#             print(
#                 f"  Skipping comparison between {algo1} and {target_algo} in {scenario} due to insufficient data."
#             )
#             continue

#         # Perform Wilcoxon Rank-Sum Test (Mann-Whitney U test)
#         u_statistic, p_value = ranksums(data_algo1, data_algo2, alternative="greater")

#         result_row = {
#             "scenario": scenario,
#             "algo1": algo1,
#             "algo2": target_algo,
#             "u_statistic": u_statistic,
#             "p_value": p_value,
#             "significant": "Yes" if p_value < alpha else "No",
#         }
#         wilcoxon_results.append(result_row)

#         print(f"  Comparing {algo1} vs {target_algo}:")
#         print(f"    U-statistic = {u_statistic:.2f}, P-value = {p_value:.4f}")
#         if p_value < alpha:
#             print(f"    Result: Significant difference (p < {alpha})")
#         else:
#             print(f"    Result: No significant difference (p >= {alpha})")

#     print("-" * 30)

# # Convert results to a DataFrame for better viewing
# wilcoxon_results_df = pd.DataFrame(wilcoxon_results)

# print("\n\n--- Summary of All Wilcoxon Test Results ---")
# print(wilcoxon_results_df)
# with open(f"{root_analysis_path}/{plot_name}.txt", "w") as f:
#     f.write(str(wilcoxon_results_df))
# wilcoxon_results_df.to_csv(f"{root_analysis_path}/wilcoxon_results.csv", index=False)

# for row in wilcoxon_results_df.itertuples():
#     signigicant = row.significant
#     scenario = row.scenario
#     algo1 = row.algo1
#     algo2 = row.algo2
#     algo1_fit = last_gen_df.query(f"algo == '{algo1}' and scenario == '{scenario}'")[
#         "mean"
#     ].item()
#     algo2_fit = last_gen_df.query(f"algo == '{algo2}' and scenario == '{scenario}'")[
#         "mean"
#     ].item()
#     if signigicant == "Yes":
#         if algo1_fit > algo2_fit:
#             pivot_table.loc[algo1, scenario] = (
#                 rf"{pivot_table.loc[algo1, scenario]}{{\bf(+)}}"
#             )
#         elif algo1_fit < algo2_fit:
#             pivot_table.loc[algo1, scenario] = (
#                 rf"{pivot_table.loc[algo1, scenario]}{{\bf(--)}}"
#             )
#     elif signigicant == "No":
#         pivot_table.loc[algo1, scenario] = (
#             rf"{pivot_table.loc[algo1, scenario]}{{($\approx$)}}"
#         )

# pivot_table.index.name = "Algorithms"
# latex_string = pivot_table.to_latex(float_format="%.1f")
# print(latex_string)
# with open(f"{root_analysis_path}/{plot_name}.txt", "a") as f:
#     f.write(latex_string)


# mean_pivot_table = last_gen_df.pivot_table(
#     index="algo",
#     columns="scenario",
#     values="mean",
#     aggfunc="first",  # Use 'first' since each scenario-algo pair has only one value
# )
# # friedmanchisquare(mean_pivot_table[])
# print(mean_pivot_table)
# stat, p = friedmanchisquare(
#     *[mean_pivot_table.T[algo].to_list() for algo in mean_pivot_table.T.columns]
# )
# print(f"\nFriedman test statistic = {stat:.4f}, p-value = {p:.4f}")
