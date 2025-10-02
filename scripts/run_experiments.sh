
#!/bin/bash

# Run all experiments for a dataset

DATASET=${1:-Cora}
DEVICE=${2:-cuda}

echo "Running all experiments on $DATASET"
echo "======================================"

# 1. Baseline experiment
echo ""
echo "1. Running baseline experiment..."
python experiments/run_baseline.py --dataset $DATASET --device $DEVICE

# 2. Dimensionality experiment
echo ""
echo "2. Running dimensionality experiment..."
python experiments/run_dimensionality.py --dataset $DATASET --device $DEVICE

# 3. GNN comparison
echo ""
echo "3. Running GNN comparison..."
python experiments/run_all_gnns.py --dataset $DATASET --device $DEVICE

# 4. Generate plots
echo ""
echo "4. Generating plots..."
python scripts/generate_plots.py

echo ""
echo "All experiments completed!"
echo "Results saved in results/metrics/"
echo "Plots saved in results/plots/"

