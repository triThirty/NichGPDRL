from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root_analysis_path = "./record"
root_data_path = "./data"

path = Path(root_data_path)

dir_list = [p for p in path.glob("*") if p.is_dir()]


plt.figure(figsize=(10, 6))

all_index_list = {}
for dir_name in dir_list:
    experiment_name = dir_name.parts[1]
    files = [f for f in dir_name.rglob("scenario_*/*_proportion_index.txt")]

    all_index_list[experiment_name] = []
    for run, file in enumerate(files):
        with open(file, "r") as f:
            a = f.readlines()
            for line in a:
                indexes = list(map(int, line.strip("[]\n").split(", ")))
                all_index_list[experiment_name].extend(indexes)

for experiment_name, index_list in all_index_list.items():
    sns.histplot(
        index_list[::],
        bins=150,
        element="step",
        fill=False,
        alpha=0.5,
        label=experiment_name,
    )

plt.legend()
plt.show()
