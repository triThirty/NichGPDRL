# Self-Attention-Guided Genetic Programming for Dynamic Scheduling: Leveraging BERT for Enhanced Tree-Structured Data Operations

This repository provides the official implementation for research into surrogate-assisted Genetic Programming (GP). By integrating **BERT-based Transformer architectures**, this approach enables effective semantic learning and feature extraction from tree-structured programs, significantly improving the efficiency of evolutionary search in dynamic scheduling environments.

---

## Project Architecture

The codebase is modularized to support experimental flexibility and comparative analysis:

* **`MTGP/`**: The core Multi-Task Genetic Programming framework, serving as the primary evolutionary engine.
* **`MTGP_KNN/`**: Implementation of the KNN-GP surrogate model, providing a baseline for localized surrogate-assisted evolution.
* **`TransformerMTGP/`**: The **BERT-SSGP** module. This implementation utilizes self-attention mechanisms to learn genotypic representations of GP trees for advanced surrogate modeling.
* **`conf/`**: Configuration management using [OmegaConf](https://omegaconf.readthedocs.io/), allowing for seamless hyperparameter tuning and experiment tracking.
* **`util/`**: Shared library components, including data structures for GP trees and logging utilities.
* **`analysis/`**: Post-processing and visualization suite to evaluate algorithm performance.

---

## Setup and Installation

### Prerequisites

- **Python**: `3.12.3`

### Installation

1. Clone the repository to your local machine.
2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Experiments are managed via `main.py` using Hydra-style parameter injection.

```bash
python main.py seeds=${selected_seed} \
               algo=${selected_algo} \
               scenarios=${selected_scenario} \
               exploration_ratio=${selected_exploration_ratio} \
               path_surfix=${selected_exploration_ratio}
```

### Configuration Arguments

| Parameter | Description | Options |
| :--- | :--- | :--- |
| `seeds` | Random seed for stochastic reproducibility | Integer |
| `algo` | The GP surrogate model variant | `MTGP`, `KNN`, `Trasformer_SSGP`, `test` |
| `scenarios` | Scheduling complexity workloads | `HH`, `HL`, `LH`, `LL` |
| `exploration_ratio` | Balance between exploitation and exploration | Float (Default: `0.8`) |
| `path_surfix` | Directory identifier for experiment logs | String |

**Data Persistence:** Results are exported to `./data/${algo}_${path_surfix}/scenario_${scenarios}`.

---

## Result Analysis and Visualization

To generate performance comparisons and benchmarking plots:

1. **Prepare Data**: Move experimental outputs to the analysis module:
    ```bash
    cp -r ./data/ ./analysis/data/
    ```

2. **Execute Benchmarking**: Run the analysis script. This procedure automatically computes metrics against the `GP` and `KNN_GP` baselines:
    ```bash
    cd analysis
    python overall_analysis.py "${plot_name}" KNN_SSGP
    ```

### Output Artifacts
- **File Format**: Analytical plots are generated in `.pdf` format.
- **Storage**: All output summaries, statistical tables, and visualizations are organized within the `record/` directory.