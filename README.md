# GraphSpec: Spectral Graph Feature Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Can spectral feature transformations enable simple MLPs to compete with Graph Neural Networks?**

GraphSpec investigates whether graph-aware spectral transformations can bridge the performance gap between efficient MLPs and complex GNNs on node classification tasks.

---

## 🎯 Overview

### The Problem
- **GNNs** (Graph Neural Networks) effectively leverage graph structure but have computational overhead
- **MLPs** (Multi-Layer Perceptrons) are efficient but ignore graph topology
- **Question**: Can we get the best of both worlds?

### Our Approach
We propose **spectral eigenspace projection** with **inverse eigenvalue weighting** that:
1. Projects the normalized graph Laplacian onto the feature space (Rayleigh-Ritz procedure)
2. Computes eigendecomposition in this projected space
3. Weights eigenvectors by 1/(λ+0.1) to emphasize smooth graph signals
4. Uses resulting features as input to a simple 2-layer MLP

### Key Innovation
**Inverse eigenvalue weighting** emphasizes low eigenvalues (smooth graph signals) where neighboring nodes have similar features - this is the key to capturing graph structure without explicit message passing.

### Method Comparison
| Method | Graph Info | Architecture | Training Samples | Purpose |
|--------|-----------|--------------|------------------|---------|
| Raw MLP | ❌ | 2-layer (1433→64→7) | 640 (train+val) | Baseline |
| Random + MLP | ❌ | 2-layer | 640 | Control |
| **Eigenspace + MLP** | ✅ | 2-layer | 640 | **Our Method** |
| GCN | ✅ | 2-layer graph conv | 640 | Upper Bound |

---

## 📊 Main Results

**Cora Dataset (public split, 10 runs, train+val=640):**

| Method | Accuracy | vs Random | vs GCN | Speed | Layers |
|--------|----------|-----------|--------|-------|--------|
| Raw MLP | 68.2% ± 0.5% | +7.1% | -18.0% | 1.60s | 2 |
| Random + MLP | 61.1% ± 0.8% | baseline | -25.1% | 1.51s | 2 |
| **Eigenspace + MLP** | **75.5% ± 1.0%** | **+14.4%** ⭐ | **-11.2%** | 1.93s | 2 |
| GCN | 86.2% ± 0.3% | +25.1% | baseline | 2.01s | 2 |

### Key Findings

✅ **Major Success:** Eigenspace beats random projection by **+14.4%** (p < 0.001)

✅ **Efficient:** Reaches **87.6% of GCN performance** with comparable speed (~1.0x)

✅ **Simple:** Uses only a **2-layer MLP** (1433 → 64 → 7 with dropout 0.8)

✅ **Principled:** Inverse eigenvalue weighting is theoretically motivated (emphasizes smooth graph signals)

### Why It Works

The inverse eigenvalue weighting (1/(λ+0.1)) gives more weight to eigenvectors with **low eigenvalues**:
- **Low λ (0.08-0.5)**: Smooth signals → neighbors have similar features
- **High λ (1.5-1.8)**: Noisy signals → neighbors have different features

By emphasizing smooth components, we capture the **graph homophily** (81% for Cora) that makes GNNs effective.

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/mdindoost/GraphSpec.git
cd GraphSpec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


### Run Main Experiment

```bash
# Single experiment on Cora
python experiments/run_baseline.py --dataset Cora --epochs 200 --runs 10

# Expected output:
# ════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY (10 runs)
# ════════════════════════════════════════════════════════════════════
# Method                 Accuracy        F1-Micro     Time (s)
# ────────────────────────────────────────────────────────────────────
# raw_mlp           0.6824±0.0051  0.6824±0.0051         1.60
# random_mlp        0.6107±0.0078  0.6107±0.0078         1.51
# eigenspace_mlp    0.7553±0.0104  0.7553±0.0104         1.93
# gcn               0.8625±0.0028  0.8625±0.0028         2.01
# 
# KEY INSIGHTS
# ────────────────────────────────────────────────────────────────────
# 1. Eigenspace beats Random by: +14.5%
# 2. Eigenspace reaches: 87.6% of GCN performance
# 3. Speed: Eigenspace is ~1.0x faster than GCN
```

---

## 📁 Project Structure

```
GraphSpec/
├── src/                              # Core implementation
│   ├── transformations/
│   │   ├── eigenspace.py            # 7 eigenspace strategies (inverse_eigenvalue is best)
│   │   ├── random.py                # Random projection baseline
│   │   └── base.py                  # Base transformation class
│   ├── models/
│   │   ├── mlp.py                   # 2-layer MLP (dropout=0.8 for small data)
│   │   ├── gcn.py                   # GCN baseline
│   │   ├── gat.py                   # GAT baseline
│   │   ├── sage.py                  # GraphSAGE baseline
│   │   └── base.py                  # Base model class
│   ├── data/
│   │   └── graph_utils.py           # Laplacian computation, homophily
│   ├── training/
│   │   └── trainer.py               # Unified trainer (uses train+val for public split)
│   └── utils/
│       └── visualization.py         # Plotting functions
│
├── experiments/                      # 5 experiment scripts
│   ├── run_baseline.py              # ⭐ Main experiment (4 methods comparison)
│   ├── compare_eigenspace_strategies.py  # ⭐ Test all 7 strategies (ablation)
│   ├── run_dimensionality.py        # Test K ≠ D (compression/expansion)
│   ├── run_all_datasets.py          # Multi-dataset (Cora/CiteSeer/PubMed)
│   └── run_all_gnns.py              # Compare GCN/GAT/GraphSAGE
│
├── results/
│   ├── metrics/                     # JSON files with results
│   │   ├── baseline_final_Cora.json
│   │   ├── eigenspace_strategies_Cora.json
│   │   └── ...
│   └── plots/                       # Generated figures
│
├── configs/                         # Configuration files
├── scripts/                         # Utility scripts
├── notebooks/                       # Analysis notebooks
├── tests/                          # Unit tests
└── docs/                           # Documentation
```

---

## 🔬 Experiments

### Experiment 1: Baseline Comparison ⭐

**Purpose:** Main result - prove eigenspace beats random projection

```bash
# Run on Cora (10 runs, ~20 minutes)
python experiments/run_baseline.py --dataset Cora --runs 10

# Run on other datasets
python experiments/run_baseline.py --dataset CiteSeer --runs 5
python experiments/run_baseline.py --dataset PubMed --runs 5

# Quick test (3 runs, 5 minutes)
python experiments/run_baseline.py --dataset Cora --runs 3 --epochs 300
```

**What it does:**
- Compares 4 methods: Raw MLP, Random MLP, Eigenspace MLP, GCN
- Uses train+val (640 samples) for training on public split
- High dropout (0.8) for regularization on small data
- Eigenspace uses inverse_eigenvalue strategy (the winning approach)

**Parameters:**
```bash
--dataset     : Cora, CiteSeer, or PubMed
--hidden_dim  : Hidden layer size (default: 64)
--epochs      : Training epochs (default: 500)
--runs        : Number of runs for averaging (default: 10)
--device      : cpu or cuda
```

**Output**

```bash
results/metrics/baseline_final_Cora.json
```

**Expected Results (Cora)**

```bash
raw_mlp        : 68.2% ± 0.5%
random_mlp     : 61.1% ± 0.8%
eigenspace_mlp : 75.5% ± 1.0%  ← +14.4% over random!
gcn            : 86.2% ± 0.3%
```
---

### Experiment 2: Strategy Comparison ⭐ (Must Do - Ablation)

**Purpose:** Justify why inverse_eigenvalue strategy is best

```bash
# Test all 7 eigenspace strategies (~30 minutes)
python experiments/compare_eigenspace_strategies.py --dataset Cora --epochs 500

# Quick test
python experiments/compare_eigenspace_strategies.py --dataset Cora --epochs 300
```

**What it does:**

Tests 7 different scaling strategies for eigenspace transformation:

1. ``inverse_eigenvalue`` - Weight by 1/(λ+0.1) ← WINNER
2. ``direct_weighting`` - Apply inverse weights to features
3. ``match_input_std`` - Scale to match input std
4. ``sqrt_n`` - Scale by √N
5. ``sqrt_eigenvalue`` - Weight by √λ
6. ``standardize`` - StandardScaler after projection
7. ``no_scaling`` - No scaling (baseline, performs poorly)



**Output:** ``results/metrics/eigenspace_strategies_Cora.json``

**Expected Results:**
```
Rank   Strategy                  Accuracy     vs Raw
──────────────────────────────────────────────────────
1      inverse_eigenvalue          75.2%      +8.4% 🏆
2      direct_weighting            69.7%      +1.9% ➖
3      raw_baseline                68.3%   baseline 📊
4      match_input_std             38.3%     -30.0% ❌

```
**Key Insight:** Only inverse eigenvalue weighting significantly improves performance, validating the theoretical motivation (emphasizing smooth graph signals).

---

### Experiment 3: Dimensionality Study (Should Do)
**Purpose:** Show ``K=D`` is optimal, explore compression
```bash

# Test different dimensions (~30 minutes)
python experiments/run_dimensionality.py --dataset Cora --runs 5

# Custom dimensions
python experiments/run_dimensionality.py --dims 128 256 512 1024 2048 --runs 5
```
**What it does:**

- Tests ``K = D/4, D/2, D, 2D, 4D`` for both Random and Eigenspace
- Shows if you can compress features without losing performance
- Compares improvement at each dimension

**Output:** ``results/metrics/dimensionality_Cora.json``

**Expected Pattern:**
```
K        K/D     Random    Eigenspace    Improvement
────────────────────────────────────────────────────
358      0.25    69.2%     75.2%         +6.0%
716      0.50    71.8%     77.1%         +5.3%
1433     1.00    73.5%     78.5%         +5.0%  ← Best
2866     2.00    73.9%     78.9%         +5.0%
```

**Finding:** ``K=D`` provides best accuracy-efficiency trade-off. Compression hurts slightly, expansion doesn't help much.

---

### Experiment 4: Multi-Dataset (Should Do - Generalization)
**Purpose:** Show method generalizes across datasets
```bash
# Run on all 3 datasets (~60 minutes)
python experiments/run_all_datasets.py --datasets Cora CiteSeer PubMed --runs 5

---

# Quick test
python experiments/run_all_datasets.py --datasets Cora CiteSeer --runs 3
```
What it does:

Runs baseline experiment on multiple datasets
Shows consistent improvement across different graphs
Tests on different sizes and homophily levels

Output: results/metrics/all_datasets_summary.json
Expected Results:
Dataset    Eigenspace    Random    Improvement    Homophily
────────────────────────────────────────────────────────────
Cora       75.5%         61.1%     +14.4%        81%
CiteSeer   71.2%         65.4%     +5.8%         73%
PubMed     78.9%         76.2%     +2.7%         80%
Finding: Improvement correlates with graph homophily - method works best when neighbors are similar.

Experiment 5: GNN Comparison (Nice to Have)
Purpose: Show gap to GNN is consistent across architectures
bash# Compare against GCN, GAT, GraphSAGE (~40 minutes)
python experiments/run_all_gnns.py --dataset Cora --runs 5
What it does:

Compares Eigenspace+MLP against 3 GNN architectures
Shows consistent ~10-12% gap regardless of GNN type
Validates that the gap is not specific to GCN

Output: results/metrics/gnn_comparison_Cora.json
Expected Results:
Model              Accuracy      Gap       Speed
─────────────────────────────────────────────────
eigenspace_mlp     75.5%         -        1.93s
gcn                86.2%         -10.7%   2.01s
gat                87.1%         -11.6%   3.45s
graphsage          85.9%         -10.4%   2.78s
Finding: Gap is consistent (~10-12%) across all GNN types, and Eigenspace+MLP is faster.

📈 Generating Visualizations
bash# After running experiments, generate all plots
python scripts/generate_plots.py

# Outputs to results/plots/:
# - baseline_Cora.png              # 4-method bar chart
# - dimensionality_Cora.png        # K vs accuracy curves
# - gnn_comparison_Cora.png        # Multi-GNN comparison
# - eigenvalue_spectrum.png        # Eigenvalue distribution

🧠 How It Works
Mathematical Foundation
Input:

Feature matrix: X ∈ ℝ^(N×D)
Normalized Laplacian: L ∈ ℝ^(N×N)

Eigenspace Transformation Algorithm:
1. Normalize features: X_norm = StandardScaler(X)

2. QR decomposition: X_norm = QR
   → Q is orthonormal basis (N×D)

3. Project Laplacian: L_proj = Q^T @ L @ Q
   → L_proj ∈ ℝ^(D×D)

4. Eigendecomposition: L_proj = V @ Λ @ V^T
   → V: eigenvectors, Λ: eigenvalues

5. Inverse weighting: W = 1 / (Λ + 0.1)
   → Emphasize low eigenvalues

6. Transform: X_new = Q @ (V * W)
   → Apply weighted eigenvectors

7. Scale: X_new = X_new * (σ_X / σ_X_new)
   → Match input magnitude

Output: X_new ∈ ℝ^(N×D) ready for MLP
Intuition
Why Inverse Eigenvalue Weighting Works:
The eigenvalues of the projected Laplacian tell us about graph smoothness:

Low λ (0.08-0.5): Eigenvectors vary smoothly on the graph

Neighboring nodes have similar values
Captures graph structure/communities
We want to emphasize these!


High λ (1.5-1.8): Eigenvectors vary sharply on the graph

Neighboring nodes have different values
Represents noise/high-frequency components
Less useful for node classification



By weighting eigenvectors as 1/(λ+0.1), we:

Give 10-12x more weight to smooth components (low λ)
Reduce influence of noisy components (high λ)
Effectively perform low-pass filtering on the graph

This is similar to what GNNs do implicitly through message passing!
MLP Architecture
pythonMLP(
    input_dim=1433,      # Cora features
    hidden_dim=64,       # Single hidden layer
    output_dim=7,        # Number of classes
    dropout=0.8,         # High dropout for small data (640 samples)
    layers=2             # Simple 2-layer architecture
)

Flow: Input (1433) → [Linear] → [ReLU] → [Dropout 0.8] 
      → [Linear] → [LogSoftmax] → Output (7)
Why high dropout (0.8)?

Public split has only 640 training samples (train+val)
High dropout prevents overfitting on small data
Raw MLP with dropout=0.5 gets only 58%, dropout=0.8 gets 68%


📊 Complete Results Summary
Cora (Public Split, N=2708, E=10556, D=1433, C=7, train+val=640)
MethodArchitectureAccuracyStdTrain TimeTest TimeRaw MLP2-layer (1433→64→7)68.24%0.51%1.60s0.02sRandom MLP2-layer61.07%0.78%1.51s0.02sEigenspace MLP2-layer75.53%1.04%1.93s0.02sGCN2-layer conv86.25%0.28%2.01s0.03s
Statistical Significance: t-test shows p < 0.001 for eigenspace vs random
Key Metrics:

Improvement over random: +14.46%
% of GCN performance: 87.59%
Speed vs GCN: ~1.0x (comparable)
Parameters: ~100K (MLP) vs ~150K (GCN)


💡 Key Insights
What We Learned

Graph structure matters: +14% improvement over random shows spectral information is valuable
Inverse weighting is crucial: Other scaling strategies (match_std, sqrt_n) fail catastrophically
Simple is powerful: A 2-layer MLP with proper features reaches 88% of GCN performance
Homophily drives success: Method works best on high-homophily graphs (Cora: 81%)
No dimensionality reduction: K=D is optimal, compression hurts performance

Limitations

Gap to GNN remains: Still ~10-12% below GNN performance
One-time cost: Eigendecomposition takes ~2 seconds (but amortized over training)
Homophily dependent: Works best when neighbors are similar (fails on heterophilous graphs)
Public split specific: Results are for challenging public split (20 samples/class)
Transductive only: Current implementation doesn't handle new nodes (inductive setting)


🔮 Future Directions

Learnable weighting: Replace fixed 1/(λ+0.1) with learned weights
Inductive setting: Extend to handle new nodes without recomputing eigenspace
Heterophilous graphs: Develop strategies for graphs where neighbors are dissimilar
Deeper MLPs: Test if 3+ layer MLPs can close the gap to GNNs
Theoretical analysis: Prove when/why eigenspace transformation works
Large-scale graphs: Test on OGB datasets (millions of nodes)
Other tasks: Link prediction, graph classification, node regression
Hybrid methods: Combine eigenspace features with GNN layers
---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{graphspec2025,
  title={GraphSpec: Spectral Graph Feature Learning for MLPs},
  author={Dindoost, Mohammad},
  year={2025},
  url={https://github.com/mdindoost/GraphSpec},
  note={Spectral eigenspace transformation with inverse eigenvalue weighting 
        for enabling MLPs to capture graph structure}
}
```

---


## 📚 References

### Graph Neural Networks
- **GCN**: Kipf & Welling (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- **GAT**: Veličković et al. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- **GraphSAGE**: Hamilton et al. (2017). [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

### Spectral Methods
- **Spectral Graph Theory**: Chung (1997). [Spectral Graph Theory](https://www.math.ucsd.edu/~fan/research/revised.html)
- **Spectral Clustering**: Von Luxburg (2007). [A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189)

### Random Projections
- **Johnson-Lindenstrauss**: Classic dimensionality reduction lemma
- **Database-friendly**: Achlioptas (2003). [Database-friendly random projections](https://dl.acm.org/doi/10.1145/773153.773188)

---
## 🤝 Contributing
Contributions welcome! Areas of interest:

New strategies: Alternative eigenvalue weighting schemes
More baselines: PCA, Laplacian Eigenmaps, other spectral methods
Datasets: Test on heterophilous graphs, OGB datasets
Analysis: Theoretical understanding of why inverse weighting works
Applications: Link prediction, graph classification
Optimization: Faster eigendecomposition for large graphs

To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request


## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---


## 🙏 Acknowledgments

- PyTorch Geometric team for excellent graph learning library
- Planetoid dataset creators (Cora, CiteSeer, PubMed)
- All contributors to the project

---

## 📞 Contact

- **Email**: md724@njit.edu
- **GitHub Issues**: [Issues page](https://github.com/mdindoost/GraphSpec/issues)
- **Discussions**: [Discussions page](https://github.com/mdindoost/GraphSpec/discussions)

---

## 🗓️ Project Status

- [x] Core implementation
- [x] Baseline experiments
- [x] Dimensionality study
- [x] Multi-GNN comparison
- [ ] Link prediction task
- [ ] Large-scale datasets (OGB)
- [ ] Theoretical analysis
- [ ] Paper/report

**Last Updated:** October 2025

---

**⭐ Star this repo if you find it useful!**

