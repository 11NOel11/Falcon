#!/usr/bin/env python3
"""
FALCON v2 Results Plotting and Analysis
Generates paper-ready figures and tables from runs/*/metrics.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def load_run(exp_name):
    """Load metrics.csv for a given experiment."""
    csv_path = f"runs/{exp_name}/metrics.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    # Filter to epoch-end rows (step == -1)
    df_epoch = df[df['step'] == -1].copy()
    return df_epoch

def main():
    os.makedirs("results", exist_ok=True)

    # Load all runs
    runs = {
        'A1_full': load_run('A1_full'),
        'M1_full_lr100': load_run('M1_full_lr100'),
        'M1_full_lr125': load_run('M1_full_lr125'),
        'F1_fast': load_run('F1_fast'),
        'A1_noise': load_run('A1_noise'),
        'F1_noise': load_run('F1_noise'),
        'A1_20p': load_run('A1_20p'),
        'F1_20p': load_run('F1_20p'),
        'A1_t10': load_run('A1_t10'),
        'M1_t10': load_run('M1_t10'),
        'F1_t10': load_run('F1_t10'),
    }

    # Select best Muon LR
    best_muon_lr = '100'
    if runs['M1_full_lr100'] is not None and runs['M1_full_lr125'] is not None:
        best_100 = runs['M1_full_lr100']['val_acc'].max()
        best_125 = runs['M1_full_lr125']['val_acc'].max()
        if best_125 > best_100:
            best_muon_lr = '125'
        print(f"Best Muon LR: {best_muon_lr} (100: {best_100:.2f}%, 125: {best_125:.2f}%)")

    # Use best Muon for plots
    runs['M1_full'] = runs[f'M1_full_lr{best_muon_lr}']

    # === FIGURE 1: Top-1 vs Wall Time ===
    print("\nGenerating fig_top1_vs_time.png...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, label, color in [
        ('A1_full', 'AdamW', 'C0'),
        ('M1_full', 'Muon (Hybrid)', 'C1'),
        ('F1_fast', 'FALCON v2', 'C2'),
    ]:
        if runs[name] is not None:
            df = runs[name]
            ax.plot(df['wall_min'], df['val_acc'], label=label, color=color, linewidth=2)

    ax.set_xlabel('Wall-clock Time (minutes)', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-10 VGG11: Top-1 Accuracy vs Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fig_top1_vs_time.png', dpi=150)
    plt.close()

    # === FIGURE 2: Time to 85% ===
    print("Generating fig_time_to_85.png...")
    fig, ax = plt.subplots(figsize=(7, 5))

    time_to_85 = {}
    for name, label in [('A1_full', 'AdamW'), ('M1_full', 'Muon'), ('F1_fast', 'FALCON v2')]:
        if runs[name] is not None:
            df = runs[name]
            idx = df[df['val_acc'] >= 85.0].index
            if len(idx) > 0:
                time_to_85[label] = df.loc[idx[0], 'wall_min']
            else:
                time_to_85[label] = None

    labels = list(time_to_85.keys())
    times = [time_to_85[l] if time_to_85[l] is not None else 0 for l in labels]
    colors = ['C0', 'C1', 'C2']

    ax.bar(labels, times, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Time to 85% Acc (minutes)', fontsize=12)
    ax.set_title('Time to Reach 85% Validation Accuracy', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fig_time_to_85.png', dpi=150)
    plt.close()

    # === FIGURE 3: Fixed-time Top-1@10min ===
    print("Generating fig_fixed_time_10min.png...")
    fig, ax = plt.subplots(figsize=(7, 5))

    acc_at_10min = {}
    for name, label in [('A1_t10', 'AdamW'), ('M1_t10', 'Muon'), ('F1_t10', 'FALCON v2')]:
        if runs[name] is not None:
            df = runs[name]
            # Get final accuracy (last epoch within budget)
            acc_at_10min[label] = df['val_acc'].iloc[-1] if len(df) > 0 else 0
        else:
            acc_at_10min[label] = 0

    labels = list(acc_at_10min.keys())
    accs = [acc_at_10min[l] for l in labels]
    colors = ['C0', 'C1', 'C2']

    ax.bar(labels, accs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Top-1 Accuracy @ 10-Minute Budget', fontsize=14)
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fig_fixed_time_10min.png', dpi=150)
    plt.close()

    # === FIGURE 4: Robustness (Clean vs Noise) ===
    print("Generating fig_robustness_noise.png...")
    fig, ax = plt.subplots(figsize=(7, 5))

    robustness = {}
    for clean_name, noise_name, label in [
        ('A1_full', 'A1_noise', 'AdamW'),
        ('F1_fast', 'F1_noise', 'FALCON v2'),
    ]:
        if runs[clean_name] is not None and runs[noise_name] is not None:
            clean_acc = runs[clean_name]['val_acc'].max()
            noise_acc = runs[noise_name]['val_acc'].max()
            delta = clean_acc - noise_acc
            robustness[label] = {'clean': clean_acc, 'noise': noise_acc, 'delta': delta}

    labels = list(robustness.keys())
    deltas = [robustness[l]['delta'] for l in labels]
    colors = ['C0', 'C2']

    ax.bar(labels, deltas, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy Drop (Clean - Noisy) (%)', fontsize=12)
    ax.set_title('Robustness: Accuracy Drop with High-Freq Noise', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fig_robustness_noise.png', dpi=150)
    plt.close()

    # === FIGURE 5: Data-Efficiency ===
    print("Generating fig_data_efficiency.png...")
    fig, ax = plt.subplots(figsize=(8, 5))

    data_eff = {}
    for name, frac, label in [
        ('A1_full', 1.0, 'AdamW 100%'),
        ('A1_20p', 0.2, 'AdamW 20%'),
        ('F1_fast', 1.0, 'FALCON 100%'),
        ('F1_20p', 0.2, 'FALCON 20%'),
    ]:
        if runs[name] is not None:
            data_eff[label] = {'frac': frac, 'acc': runs[name]['val_acc'].max()}

    adamw_fracs = [data_eff['AdamW 100%']['frac'], data_eff['AdamW 20%']['frac']]
    adamw_accs = [data_eff['AdamW 100%']['acc'], data_eff['AdamW 20%']['acc']]
    falcon_fracs = [data_eff['FALCON 100%']['frac'], data_eff['FALCON 20%']['frac']]
    falcon_accs = [data_eff['FALCON 100%']['acc'], data_eff['FALCON 20%']['acc']]

    ax.plot(adamw_fracs, adamw_accs, 'o-', label='AdamW', color='C0', linewidth=2, markersize=8)
    ax.plot(falcon_fracs, falcon_accs, 's-', label='FALCON v2', color='C2', linewidth=2, markersize=8)

    ax.set_xlabel('Training Data Fraction', fontsize=12)
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax.set_title('Data Efficiency: Accuracy vs Training Data', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.1])
    plt.tight_layout()
    plt.savefig('results/fig_data_efficiency.png', dpi=150)
    plt.close()

    # === SUMMARY TABLE ===
    print("\nGenerating table_summary.csv...")
    summary_data = []

    for name, label in [
        ('A1_full', 'AdamW'),
        ('M1_full', f'Muon (LR={best_muon_lr})'),
        ('F1_fast', 'FALCON v2'),
    ]:
        if runs[name] is not None:
            df = runs[name]
            best_acc = df['val_acc'].max()
            best_epoch = df.loc[df['val_acc'].idxmax(), 'epoch']
            avg_epoch_time = df['wall_min'].diff().mean() if len(df) > 1 else 0

            # Time to 85%
            idx_85 = df[df['val_acc'] >= 85.0].index
            time_85 = df.loc[idx_85[0], 'wall_min'] if len(idx_85) > 0 else None

            summary_data.append({
                'Optimizer': label,
                'Best Val Acc (%)': f"{best_acc:.2f}",
                'Best Epoch': int(best_epoch),
                'Avg Time/Epoch (min)': f"{avg_epoch_time:.2f}" if avg_epoch_time > 0 else "N/A",
                'Time to 85% (min)': f"{time_85:.2f}" if time_85 is not None else "N/A",
            })

    # Add fixed-time results
    for name, label in [('A1_t10', 'AdamW @ 10min'), ('M1_t10', 'Muon @ 10min'), ('F1_t10', 'FALCON @ 10min')]:
        if runs[name] is not None:
            df = runs[name]
            final_acc = df['val_acc'].iloc[-1] if len(df) > 0 else 0
            final_epoch = df['epoch'].iloc[-1] if len(df) > 0 else 0
            summary_data.append({
                'Optimizer': label,
                'Best Val Acc (%)': f"{final_acc:.2f}",
                'Best Epoch': int(final_epoch),
                'Avg Time/Epoch (min)': "N/A",
                'Time to 85% (min)': "N/A",
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/table_summary.csv', index=False)

    # === RESULTS MARKDOWN ===
    print("\nGenerating results.md...")
    with open('results/results.md', 'w') as f:
        f.write("# FALCON v2: Experimental Results Summary\n\n")
        f.write(f"**Best Muon LR Multiplier:** {best_muon_lr} (1.{best_muon_lr}x base LR)\n\n")

        f.write("## Primary Results (60 Epochs, Full Data)\n\n")
        f.write("| Optimizer | Best Val Acc (%) | Epoch | Avg Time/Epoch (min) | Time to 85% (min) |\n")
        f.write("|-----------|------------------|-------|----------------------|-------------------|\n")

        for name, label in [('A1_full', 'AdamW'), ('M1_full', f'Muon'), ('F1_fast', 'FALCON v2')]:
            if runs[name] is not None:
                df = runs[name]
                best_acc = df['val_acc'].max()
                best_epoch = int(df.loc[df['val_acc'].idxmax(), 'epoch'])
                avg_time = df['wall_min'].diff().mean() if len(df) > 1 else 0
                idx_85 = df[df['val_acc'] >= 85.0].index
                time_85 = df.loc[idx_85[0], 'wall_min'] if len(idx_85) > 0 else None
                f.write(f"| {label} | {best_acc:.2f} | {best_epoch} | {avg_time:.2f} | ")
                f.write(f"{time_85:.2f} |\n" if time_85 is not None else "N/A |\n")

        f.write("\n## Fixed-Time Results (10-Minute Budget)\n\n")
        f.write("| Optimizer | Final Val Acc (%) | Epochs Completed |\n")
        f.write("|-----------|-------------------|------------------|\n")

        for name, label in [('A1_t10', 'AdamW'), ('M1_t10', 'Muon'), ('F1_t10', 'FALCON v2')]:
            if runs[name] is not None:
                df = runs[name]
                final_acc = df['val_acc'].iloc[-1] if len(df) > 0 else 0
                final_epoch = int(df['epoch'].iloc[-1]) if len(df) > 0 else 0
                f.write(f"| {label} | {final_acc:.2f} | {final_epoch} |\n")

        f.write("\n## Robustness (High-Freq Noise σ=0.15)\n\n")
        f.write("| Optimizer | Clean Acc (%) | Noisy Acc (%) | Accuracy Drop (%) |\n")
        f.write("|-----------|---------------|---------------|-------------------|\n")

        for clean_name, noise_name, label in [('A1_full', 'A1_noise', 'AdamW'), ('F1_fast', 'F1_noise', 'FALCON v2')]:
            if runs[clean_name] is not None and runs[noise_name] is not None:
                clean_acc = runs[clean_name]['val_acc'].max()
                noise_acc = runs[noise_name]['val_acc'].max()
                delta = clean_acc - noise_acc
                f.write(f"| {label} | {clean_acc:.2f} | {noise_acc:.2f} | {delta:.2f} |\n")

        f.write("\n## Data Efficiency (20% Training Data)\n\n")
        f.write("| Optimizer | 100% Data Acc (%) | 20% Data Acc (%) | Accuracy Drop (%) |\n")
        f.write("|-----------|-------------------|------------------|-------------------|\n")

        for full_name, frac_name, label in [('A1_full', 'A1_20p', 'AdamW'), ('F1_fast', 'F1_20p', 'FALCON v2')]:
            if runs[full_name] is not None and runs[frac_name] is not None:
                full_acc = runs[full_name]['val_acc'].max()
                frac_acc = runs[frac_name]['val_acc'].max()
                delta = full_acc - frac_acc
                f.write(f"| {label} | {full_acc:.2f} | {frac_acc:.2f} | {delta:.2f} |\n")

        f.write("\n## Figures\n\n")
        f.write("- `fig_top1_vs_time.png`: Top-1 accuracy vs wall-clock time\n")
        f.write("- `fig_time_to_85.png`: Time to reach 85% accuracy\n")
        f.write("- `fig_fixed_time_10min.png`: Top-1 accuracy at 10-minute budget\n")
        f.write("- `fig_robustness_noise.png`: Accuracy drop with high-frequency noise\n")
        f.write("- `fig_data_efficiency.png`: Accuracy vs training data fraction\n")

    print("\n✓ All plots and tables generated in results/")
    print("\nFiles created:")
    print("  - fig_top1_vs_time.png")
    print("  - fig_time_to_85.png")
    print("  - fig_fixed_time_10min.png")
    print("  - fig_robustness_noise.png")
    print("  - fig_data_efficiency.png")
    print("  - table_summary.csv")
    print("  - results.md")

if __name__ == '__main__':
    main()
