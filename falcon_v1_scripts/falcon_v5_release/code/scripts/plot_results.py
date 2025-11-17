"""
FALCON v3 Results Plotting and Analysis
Generates paper-quality figures and summary table from experiment runs.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure matplotlib for paper-quality plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs")

# Experiment groups
EXP_GROUPS = {
    "full": ["A1_full", "M1_full_lr100", "M1_full_lr125", "F1_v3"],
    "noise": ["A1_noise", "M1_noise", "F1_noise"],
    "data20p": ["A1_20p", "M1_20p", "F1_20p"],
    "time10min": ["A1_t10", "M1_t10", "F1_t10"],
}

# Display names
DISPLAY_NAMES = {
    "A1_full": "AdamW",
    "M1_full_lr100": "Muon (1.0x)",
    "M1_full_lr125": "Muon (1.25x)",
    "F1_v3": "FALCON v3",
    "A1_noise": "AdamW",
    "M1_noise": "Muon",
    "F1_noise": "FALCON v3",
    "A1_20p": "AdamW",
    "M1_20p": "Muon",
    "F1_20p": "FALCON v3",
    "A1_t10": "AdamW",
    "M1_t10": "Muon",
    "F1_t10": "FALCON v3",
}

# Colors
COLORS = {
    "AdamW": "#1f77b4",
    "Muon (1.0x)": "#ff7f0e",
    "Muon (1.25x)": "#d62728",
    "Muon": "#d62728",
    "FALCON v3": "#2ca02c",
}


def load_metrics(exp_name):
    """Load metrics CSV for an experiment."""
    csv_path = RUNS_DIR / exp_name / "metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def get_time_to_target(df, target_acc=85.0):
    """Get time (minutes) to reach target accuracy."""
    if df is None or 'val_acc' not in df.columns:
        return None
    mask = df['val_acc'] >= target_acc
    if not mask.any():
        return None
    return df.loc[mask, 'wall_min'].iloc[0]


def get_acc_at_time(df, target_time=10.0):
    """Get best accuracy reached within target time (minutes)."""
    if df is None or 'wall_min' not in df.columns:
        return None
    mask = df['wall_min'] <= target_time
    if not mask.any():
        return None
    return df.loc[mask, 'val_acc'].max()


def plot_top1_vs_time():
    """Plot Top-1 accuracy vs wall clock time for full training."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use best Muon variant
    exps_to_plot = ["A1_full", "M1_full_lr125", "F1_v3"]
    
    for exp in exps_to_plot:
        df = load_metrics(exp)
        if df is not None and 'wall_min' in df.columns and 'val_acc' in df.columns:
            label = DISPLAY_NAMES.get(exp, exp)
            color = COLORS.get(label, None)
            ax.plot(df['wall_min'], df['val_acc'], label=label, color=color, marker='o', markevery=5)
    
    ax.set_xlabel('Wall Clock Time (minutes)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Convergence: Top-1 vs Time (CIFAR-10, VGG11)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    output_path = RESULTS_DIR / "fig_top1_vs_time.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_time_to_85():
    """Bar plot: time to reach 85% accuracy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    exps = ["A1_full", "M1_full_lr125", "F1_v3"]
    labels = []
    times = []
    colors_list = []
    
    for exp in exps:
        df = load_metrics(exp)
        t = get_time_to_target(df, target_acc=85.0)
        if t is not None:
            label = DISPLAY_NAMES.get(exp, exp)
            labels.append(label)
            times.append(t)
            colors_list.append(COLORS.get(label, 'gray'))
    
    if times:
        bars = ax.bar(labels, times, color=colors_list, alpha=0.8)
        ax.set_ylabel('Time to 85% Accuracy (minutes)')
        ax.set_title('Time to Reach 85% Top-1 (Lower is Better)')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{t:.1f}',
                   ha='center', va='bottom', fontsize=10)
    
    output_path = RESULTS_DIR / "fig_time_to_85.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fixed_time_10min():
    """Bar plot: Top-1 at 10 minutes fixed time budget."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    exps = ["A1_t10", "M1_t10", "F1_t10"]
    labels = []
    accs = []
    colors_list = []
    
    for exp in exps:
        df = load_metrics(exp)
        acc = get_acc_at_time(df, target_time=10.0)
        if acc is not None:
            label = DISPLAY_NAMES.get(exp, exp)
            labels.append(label)
            accs.append(acc)
            colors_list.append(COLORS.get(label, 'gray'))
    
    if accs:
        bars = ax.bar(labels, accs, color=colors_list, alpha=0.8)
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('Fixed-Time Fairness: Top-1 @ 10 Minutes')
        ax.set_ylim([min(accs)-2, max(accs)+2])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}',
                   ha='center', va='bottom', fontsize=10)
    
    output_path = RESULTS_DIR / "fig_fixed_time_10min.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_robustness_noise():
    """Bar plot: Delta clean vs noisy accuracy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Get clean accuracies
    clean_exps = ["A1_full", "M1_full_lr125", "F1_v3"]
    noise_exps = ["A1_noise", "M1_noise", "F1_noise"]
    
    labels = []
    deltas = []
    colors_list = []
    
    for clean_exp, noise_exp in zip(clean_exps, noise_exps):
        df_clean = load_metrics(clean_exp)
        df_noise = load_metrics(noise_exp)
        
        if df_clean is not None and df_noise is not None:
            clean_acc = df_clean['val_acc'].max()
            noise_acc = df_noise['val_acc'].max()
            delta = clean_acc - noise_acc
            
            label = DISPLAY_NAMES.get(noise_exp, noise_exp)
            labels.append(label)
            deltas.append(delta)
            colors_list.append(COLORS.get(label, 'gray'))
    
    if deltas:
        bars = ax.bar(labels, deltas, color=colors_list, alpha=0.8)
        ax.set_ylabel('Accuracy Drop (% points)')
        ax.set_title('Robustness: Clean - Noisy Accuracy (σ=0.15)')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{delta:.1f}',
                   ha='center', va='bottom', fontsize=10)
    
    output_path = RESULTS_DIR / "fig_robustness_noise.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_data_efficiency():
    """Bar plot: Accuracy at 20% data vs full data."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    full_exps = ["A1_full", "M1_full_lr125", "F1_v3"]
    data20_exps = ["A1_20p", "M1_20p", "F1_20p"]
    
    labels = []
    accs_full = []
    accs_20p = []
    colors_list = []
    
    for full_exp, data20_exp in zip(full_exps, data20_exps):
        df_full = load_metrics(full_exp)
        df_20p = load_metrics(data20_exp)
        
        if df_full is not None and df_20p is not None:
            acc_full = df_full['val_acc'].max()
            acc_20p = df_20p['val_acc'].max()
            
            label = DISPLAY_NAMES.get(data20_exp, data20_exp)
            labels.append(label)
            accs_full.append(acc_full)
            accs_20p.append(acc_20p)
            colors_list.append(COLORS.get(label, 'gray'))
    
    if accs_full and accs_20p:
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accs_full, width, label='Full Data (100%)', alpha=0.8)
        bars2 = ax.bar(x + width/2, accs_20p, width, label='Limited Data (20%)', alpha=0.8)
        
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('Data Efficiency: Full vs 20% Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    output_path = RESULTS_DIR / "fig_data_efficiency.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_table():
    """Generate comprehensive summary table."""
    rows = []
    
    # Full training runs
    for exp in ["A1_full", "M1_full_lr100", "M1_full_lr125", "F1_v3"]:
        df = load_metrics(exp)
        if df is not None:
            best_acc = df['val_acc'].max()
            best_epoch = df.loc[df['val_acc'].idxmax(), 'epoch']
            final_time = df['wall_min'].iloc[-1]
            avg_time_per_epoch = final_time / len(df)
            time_to_85 = get_time_to_target(df, 85.0)
            
            rows.append({
                'Experiment': DISPLAY_NAMES.get(exp, exp),
                'Best_Val_Acc': f"{best_acc:.2f}",
                'Best_Epoch': int(best_epoch),
                'Total_Time_Min': f"{final_time:.2f}",
                'Avg_Time_Per_Epoch_Sec': f"{avg_time_per_epoch*60:.1f}",
                'Time_To_85_Min': f"{time_to_85:.2f}" if time_to_85 else "N/A",
            })
    
    # Fixed time experiments
    for exp in ["A1_t10", "M1_t10", "F1_t10"]:
        df = load_metrics(exp)
        if df is not None:
            acc_at_10 = get_acc_at_time(df, 10.0)
            rows.append({
                'Experiment': f"{DISPLAY_NAMES.get(exp, exp)} @10min",
                'Best_Val_Acc': f"{acc_at_10:.2f}" if acc_at_10 else "N/A",
                'Best_Epoch': "N/A",
                'Total_Time_Min': "10.00",
                'Avg_Time_Per_Epoch_Sec': "N/A",
                'Time_To_85_Min': "N/A",
            })
    
    # Data efficiency
    for exp in ["A1_20p", "M1_20p", "F1_20p"]:
        df = load_metrics(exp)
        if df is not None:
            best_acc = df['val_acc'].max()
            rows.append({
                'Experiment': f"{DISPLAY_NAMES.get(exp, exp)} @20%",
                'Best_Val_Acc': f"{best_acc:.2f}",
                'Best_Epoch': "N/A",
                'Total_Time_Min': "N/A",
                'Avg_Time_Per_Epoch_Sec': "N/A",
                'Time_To_85_Min': "N/A",
            })
    
    # Robustness
    clean_exps = ["A1_full", "M1_full_lr125", "F1_v3"]
    noise_exps = ["A1_noise", "M1_noise", "F1_noise"]
    for clean_exp, noise_exp in zip(clean_exps, noise_exps):
        df_clean = load_metrics(clean_exp)
        df_noise = load_metrics(noise_exp)
        if df_clean is not None and df_noise is not None:
            clean_acc = df_clean['val_acc'].max()
            noise_acc = df_noise['val_acc'].max()
            delta = clean_acc - noise_acc
            rows.append({
                'Experiment': f"{DISPLAY_NAMES.get(noise_exp, noise_exp)} noise",
                'Best_Val_Acc': f"{noise_acc:.2f}",
                'Best_Epoch': "N/A",
                'Total_Time_Min': "N/A",
                'Avg_Time_Per_Epoch_Sec': "N/A",
                'Time_To_85_Min': f"Δ={delta:.2f}",
            })
    
    df_summary = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "table_summary.csv"
    df_summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))


def main():
    """Generate all plots and summary table."""
    print("=" * 60)
    print(" FALCON v3 Results Analysis")
    print("=" * 60)
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    plot_top1_vs_time()
    plot_time_to_85()
    plot_fixed_time_10min()
    plot_robustness_noise()
    plot_data_efficiency()
    
    print("\nGenerating summary table...")
    generate_summary_table()
    
    print("\n" + "=" * 60)
    print(f" All results saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
