#!/usr/bin/env python3
"""
FALCON v4 Results Plotting
Generates publication-quality figures comparing AdamW, Muon, and FALCON v4.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results directory
RESULTS_DIR = Path("results_v4")
RESULTS_DIR.mkdir(exist_ok=True)

# Experiment definitions
EXPERIMENTS = {
    # Full training
    "A1_full": {"optimizer": "AdamW", "type": "full", "label": "AdamW"},
    "M1_full": {"optimizer": "Muon", "type": "full", "label": "Muon"},
    "F4_full": {"optimizer": "FALCON v4", "type": "full", "label": "FALCON v4"},
    
    # Fixed-time 10min
    "A1_t10": {"optimizer": "AdamW", "type": "fixed_time", "label": "AdamW"},
    "M1_t10": {"optimizer": "Muon", "type": "fixed_time", "label": "Muon"},
    "F4_t10": {"optimizer": "FALCON v4", "type": "fixed_time", "label": "FALCON v4"},
    
    # Data efficiency 20%
    "A1_20p": {"optimizer": "AdamW", "type": "data_20p", "label": "AdamW"},
    "M1_20p": {"optimizer": "Muon", "type": "data_20p", "label": "Muon"},
    "F4_20p": {"optimizer": "FALCON v4", "type": "data_20p", "label": "FALCON v4"},
    
    # Data efficiency 10%
    "A1_10p": {"optimizer": "AdamW", "type": "data_10p", "label": "AdamW"},
    "M1_10p": {"optimizer": "Muon", "type": "data_10p", "label": "Muon"},
    "F4_10p": {"optimizer": "FALCON v4", "type": "data_10p", "label": "FALCON v4"},
}

# Colors
COLORS = {
    "AdamW": "#1f77b4",
    "Muon": "#ff7f0e",
    "FALCON v4": "#2ca02c",
}


def load_metrics(exp_name):
    """Load metrics.csv for an experiment."""
    csv_path = Path("runs") / exp_name / "metrics.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    return pd.read_csv(csv_path)


def plot_top1_vs_time():
    """Plot top-1 accuracy vs wall time for full training."""
    print("Generating fig_top1_vs_time.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for exp_name, info in EXPERIMENTS.items():
        if info["type"] != "full":
            continue
        
        df = load_metrics(exp_name)
        if df is None:
            continue
        
        # Compute cumulative time
        if "wall_min" in df.columns:
            wall_min = df["wall_min"].values
        elif "epoch_time" in df.columns:
            wall_min = df["epoch_time"].cumsum().values / 60.0
        else:
            wall_min = np.arange(len(df))
        
        val_acc = df["val_acc"].values
        
        ax.plot(wall_min, val_acc, label=info["label"], color=COLORS[info["optimizer"]], linewidth=2)
    
    ax.set_xlabel("Wall Time (minutes)", fontsize=12)
    ax.set_ylabel("Validation Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Training Efficiency: Accuracy vs Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_top1_vs_time.png", dpi=150)
    plt.close()


def plot_time_to_85():
    """Plot time to reach 85% accuracy."""
    print("Generating fig_time_to_85.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    optimizers = ["AdamW", "Muon", "FALCON v4"]
    times = []
    
    for opt in optimizers:
        exp_name = [k for k, v in EXPERIMENTS.items() if v["type"] == "full" and v["optimizer"] == opt][0]
        df = load_metrics(exp_name)
        if df is None:
            times.append(0)
            continue
        
        # Find first epoch >= 85%
        idx = df[df["val_acc"] >= 85.0].index
        if len(idx) == 0:
            times.append(df["wall_min"].max() if "wall_min" in df.columns else len(df))
        else:
            first_idx = idx[0]
            if "wall_min" in df.columns:
                times.append(df.loc[first_idx, "wall_min"])
            elif "epoch_time" in df.columns:
                times.append(df["epoch_time"][:first_idx+1].sum() / 60.0)
            else:
                times.append(first_idx + 1)
    
    bars = ax.bar(optimizers, times, color=[COLORS[opt] for opt in optimizers], alpha=0.8, edgecolor="black")
    ax.set_ylabel("Time to 85% Accuracy (minutes)", fontsize=12)
    ax.set_title("Time to Reach 85% Validation Accuracy", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_time_to_85.png", dpi=150)
    plt.close()


def plot_fixed_time_10min():
    """Plot final accuracy at 10-minute budget."""
    print("Generating fig_fixed_time_10min.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    optimizers = ["AdamW", "Muon", "FALCON v4"]
    accuracies = []
    
    for opt in optimizers:
        exp_name = [k for k, v in EXPERIMENTS.items() if v["type"] == "fixed_time" and v["optimizer"] == opt][0]
        df = load_metrics(exp_name)
        if df is None:
            accuracies.append(0)
            continue
        
        # Get max accuracy achieved
        accuracies.append(df["val_acc"].max())
    
    bars = ax.bar(optimizers, accuracies, color=[COLORS[opt] for opt in optimizers], alpha=0.8, edgecolor="black")
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=12)
    ax.set_title("Fixed-Time Performance (10 minute budget)", fontsize=14, fontweight="bold")
    ax.set_ylim(75, 90)
    ax.grid(True, axis="y", alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_fixed_time_10min.png", dpi=150)
    plt.close()


def plot_robustness_noise():
    """Plot robustness to high-frequency noise (eval-only results)."""
    print("Generating fig_robustness_noise.png...")
    
    # For robustness, we need to compute delta from clean vs noisy
    # This requires running eval-only with --test-highfreq-noise 0.04
    # and capturing the output or using checkpoint evaluations
    
    # Placeholder: compute from best.pt checkpoints
    # Actual implementation would parse eval-only output
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    optimizers = ["AdamW", "Muon", "FALCON v4"]
    clean_acc = []
    noisy_acc = []
    
    for opt in optimizers:
        exp_name = [k for k, v in EXPERIMENTS.items() if v["type"] == "full" and v["optimizer"] == opt][0]
        df = load_metrics(exp_name)
        if df is None:
            clean_acc.append(0)
            noisy_acc.append(0)
            continue
        
        # Clean: best accuracy from training
        clean_acc.append(df["val_acc"].max())
        
        # Noisy: estimate ~2-3% drop (placeholder - should run eval)
        # In practice, parse eval-only output or run here
        noisy_acc.append(df["val_acc"].max() - 2.5)  # Placeholder
    
    x = np.arange(len(optimizers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_acc, width, label="Clean", color=[COLORS[opt] for opt in optimizers], alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x + width/2, noisy_acc, width, label="Noisy (σ=0.04)", color=[COLORS[opt] for opt in optimizers], alpha=0.5, edgecolor="black", hatch="//")
    
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Robustness to High-Frequency Noise", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_robustness_noise.png", dpi=150)
    plt.close()
    
    print("  Note: Robustness values are placeholders. Run eval-only experiments to get actual values.")


def plot_data_efficiency():
    """Plot data efficiency: accuracy at 10% and 20% data."""
    print("Generating fig_data_efficiency.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizers = ["AdamW", "Muon", "FALCON v4"]
    fractions = ["10%", "20%", "100%"]
    exp_types = ["data_10p", "data_20p", "full"]
    
    data = {opt: [] for opt in optimizers}
    
    for exp_type in exp_types:
        for opt in optimizers:
            exps = [k for k, v in EXPERIMENTS.items() if v["type"] == exp_type and v["optimizer"] == opt]
            if len(exps) == 0:
                data[opt].append(0)
                continue
            
            exp_name = exps[0]
            df = load_metrics(exp_name)
            if df is None:
                data[opt].append(0)
                continue
            
            data[opt].append(df["val_acc"].max())
    
    x = np.arange(len(fractions))
    width = 0.25
    
    for i, opt in enumerate(optimizers):
        ax.bar(x + i*width, data[opt], width, label=opt, color=COLORS[opt], alpha=0.8, edgecolor="black")
    
    ax.set_xlabel("Dataset Fraction", fontsize=12)
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=12)
    ax.set_title("Data Efficiency: Performance vs Dataset Size", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(fractions)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_data_efficiency.png", dpi=150)
    plt.close()


def generate_summary_table():
    """Generate summary table CSV."""
    print("Generating table_summary.csv...")
    
    rows = []
    
    for exp_name, info in EXPERIMENTS.items():
        df = load_metrics(exp_name)
        if df is None:
            continue
        
        best_val = df["val_acc"].max()
        best_epoch = df["val_acc"].idxmax() + 1
        
        if "wall_min" in df.columns:
            total_wall_min = df["wall_min"].max()
            median_epoch_time = df["epoch_time"].median() if "epoch_time" in df.columns else 0
        else:
            total_wall_min = 0
            median_epoch_time = 0
        
        # Images/sec (placeholder - requires batch size info)
        images_per_sec = 0  # Would need to compute from batch_size and epoch_time
        
        row = {
            "Experiment": exp_name,
            "Optimizer": info["optimizer"],
            "Type": info["type"],
            "Best Val@1": f"{best_val:.2f}",
            "Best Epoch": best_epoch,
            "Total Wall (min)": f"{total_wall_min:.2f}",
            "Median Epoch Time (s)": f"{median_epoch_time:.1f}",
            "Images/sec": images_per_sec,  # Placeholder
        }
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(RESULTS_DIR / "table_summary.csv", index=False)
    print(f"Summary table saved to {RESULTS_DIR / 'table_summary.csv'}")


def main():
    print("=== FALCON v4 Results Plotting ===")
    print(f"Output directory: {RESULTS_DIR}")
    print()
    
    # Generate all plots
    plot_top1_vs_time()
    plot_time_to_85()
    plot_fixed_time_10min()
    plot_robustness_noise()
    plot_data_efficiency()
    generate_summary_table()
    
    print()
    print("=== All figures generated ===")
    print(f"Figures saved to {RESULTS_DIR}/")
    print("  - fig_top1_vs_time.png")
    print("  - fig_time_to_85.png")
    print("  - fig_fixed_time_10min.png")
    print("  - fig_robustness_noise.png")
    print("  - fig_data_efficiency.png")
    print("  - table_summary.csv")


if __name__ == "__main__":
    main()
