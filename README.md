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
We propose **spectral eigenspace projection** that:
1. Projects the normalized graph Laplacian onto the feature space (Rayleigh-Ritz)
2. Computes eigendecomposition in this projected space
3. Uses resulting eigenvectors as graph-aware features for MLPs

### Key Comparisons
| Method | Graph Info | Complexity | Purpose |
|--------|-----------|-----------|---------|
| Raw MLP | ❌ | Lowest | Baseline |
| Random + MLP | ❌ | Low | Control |
| **Eigenspace + MLP** | ✅ | Low | **Our Method** |
| GNN (GCN/GAT/SAGE) | ✅ | Higher | Upper Bound |

---

## 📊 Quick Results

**Cora Dataset (10 runs average):**

| Method | Accuracy | vs Random | vs GNN | Speed |

**Key Finding:** Eigenspace projection achieves...

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mdindoost/GraphSpec.git
cd GraphSpec

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Baseline Experiment

```bash
# Single experiment on Cora
python experiments/run_baseline.py --dataset Cora --epochs 200

# Expected output:
# ============================================================
# Method                  Accuracy   F1-Micro   Time (s)
# ------------------------------------------------------------
# Raw MLP                
# Random + MLP           
# Eigenspace + MLP       
# GCN                     
# ============================================================
```

### Run All Experiments

```bash
# Complete experimental pipeline
bash scripts/run_experiments.sh Cora cuda

# This runs:
# 1. Baseline comparison (K=D)
# 2. Dimensionality study (K≠D)
# 3. GNN architecture comparison
# 4. Generate plots
```

---

## 📁 Project Structure

```
GraphSpec/
├── src/                          # Core implementation
│   ├── transformations/          # Feature transformations
│   │   ├── eigenspace.py        # Spectral projection (main method)
│   │   ├── random.py            # Random projection (baseline)
│   │   └── base.py              # Base class
│   ├── models/                   # Neural network models
│   │   ├── mlp.py               # MLP architecture
│   │   ├── gcn.py, gat.py, sage.py  # GNN baselines
│   │   └── base.py              # Base model class
│   ├── data/                     # Data utilities
│   │   └── graph_utils.py       # Graph operations
│   ├── training/                 # Training pipeline
│   │   └── trainer.py           # Unified trainer
│   └── utils/                    # Utilities
│       └── visualization.py     # Plotting functions
│
├── experiments/                  # Experiment scripts
│   ├── run_baseline.py          # Main comparison (K=D)
│   ├── run_dimensionality.py    # Dimension study (K≠D)
│   ├── run_all_gnns.py          # GNN comparison
│   ├── run_all_datasets.py      # Multi-dataset
│   └── run_eigenvalue_analysis.py  # Spectral analysis
│
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── notebooks/                    # Analysis notebooks
├── results/                      # Experiment outputs
│   ├── metrics/                 # Performance metrics (JSON)
│   └── plots/                   # Generated figures
└── docs/                        # Documentation
```

---

## 🔬 Experiments

### 1. Baseline Comparison (K = D)

**Goal:** Fair comparison on equal dimensions

```bash
python experiments/run_baseline.py --dataset Cora --runs 10
```

**Tests:**
- ✓ Raw MLP (no transformation)
- ✓ Random projection + MLP (control)
- ✓ Eigenspace projection + MLP (proposed)
- ✓ GCN (gold standard)

**Answers:**
1. Does eigenspace outperform random?
2. How close to GNN performance?
3. Speed vs accuracy trade-off?

---

### 2. Dimensionality Study (K ≠ D)

**Goal:** Find optimal target dimension

```bash
python experiments/run_dimensionality.py --dataset Cora
```

**Tests:**
- K < D: Compression (e.g., D/4, D/2)
- K = D: Fair comparison
- K > D: Expansion (e.g., 2D, 4D)

**Answers:**
1. Is K=D optimal?
2. Can we reduce dimension without losing performance?
3. Does expansion help?

**Example Results:**

| K | K/D | Random | Eigenspace | Improvement |
|---|-----|--------|-----------|-------------|


**Finding:** K=D provides 

---

### 3. GNN Architecture Comparison

**Goal:** Compare against multiple GNN types

```bash
python experiments/run_all_gnns.py --dataset Cora
```

**Tests:**
- Eigenspace + MLP
- GCN (convolutional)
- GAT (attention-based)
- GraphSAGE (sampling-based)

**Answers:**
1. Consistent gap across GNN types?
2. Which GNN is closest to eigenspace?
3. Speed comparison?

---

### 4. Multi-Dataset Evaluation

**Goal:** Generalization across datasets

```bash
python experiments/run_all_datasets.py --datasets Cora CiteSeer PubMed
```

**Tests:** Baseline on multiple citation networks

**Answers:**
1. Consistent improvement across datasets?
2. When does eigenspace work best?

---

## 📈 Generating Plots

```bash
# After running experiments
python scripts/generate_plots.py

# Outputs to results/plots/:
# - baseline_{dataset}.png          # 4-method comparison
# - dimensionality_{dataset}.png    # K vs accuracy curve
# - gnn_comparison_{dataset}.png    # Multiple GNNs
```

**Example Plots:**

![Baseline Comparison](results/plots/baseline_Cora.png)
*Eigenspace significantly outperforms random projection*

![Dimensionality Analysis](results/plots/dimensionality_Cora.png)
*Performance vs dimension trade-off*

---

## 🧠 How It Works

### Mathematical Foundation

**Input:**
- Feature matrix: X ∈ ℝ^(N×D)
- Graph Laplacian: L ∈ ℝ^(N×N)

**Eigenspace Transformation:**

1. **QR Decomposition**: X = QR, get orthonormal basis Q
2. **Project Laplacian**: L_proj = Q^T L Q ∈ ℝ^(D×D)
3. **Eigendecomposition**: L_proj = V Λ V^T
4. **Transform**: X_new = Q V ∈ ℝ^(N×D)

**Result:** Features ordered by graph smoothness (eigenvalues)

**Intuition:**
- Low eigenvalues → smooth signals on graph (connected nodes similar)
- High eigenvalues → rapidly varying signals (connected nodes different)
- Reordering captures graph structure in feature space

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{graphspec,
  title={GraphSpec: Spectral Graph Feature Learning for MLPs},
  author={Mohammad Dindoost},
  year={2025},
  url={https://github.com/mdindoost/GraphSpec},
  note={Research project investigating spectral transformations for graph learning}
}
```

---

## 📝 Results Summary

### Main Findings


### Limitations


### Future Directions


---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for contribution:**
- New transformation methods
- Additional GNN baselines
- More datasets (OGB, heterogeneous graphs)
- Theoretical analysis
- Optimization improvements
- Documentation

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

