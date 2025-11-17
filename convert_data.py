#!/usr/bin/env python3
"""
Convert Falcon v5 experimental CSV data to JSON format for the UI
"""

import csv
import json
import os
from pathlib import Path

# Paths
RESULTS_DIR = Path("falcon_v5_release/results")
OUTPUT_DIR = Path("falcon_ui/data")

def parse_csv(filepath):
    """Parse a CSV file and return the data as a list of dictionaries"""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def get_epoch_data(csv_data):
    """Extract epoch-level data (step == -1 indicates end of epoch)"""
    epoch_data = []
    for row in csv_data:
        if row['step'] == '-1':
            epoch_data.append({
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'val_acc': float(row['val_acc']) if row['val_acc'] else None,
                'epoch_time': float(row['epoch_time']) if row['epoch_time'] else None,
                'imgs_per_sec': float(row['imgs_per_sec']) if row['imgs_per_sec'] else None,
                'wall_min': float(row['wall_min'])
            })
    return epoch_data

def create_dynamics_json():
    """Create the dynamics.json file with real training data"""

    # Load the three main experiments
    adamw_data = parse_csv(RESULTS_DIR / "A1_full_metrics.csv")
    muon_data = parse_csv(RESULTS_DIR / "M1_full_metrics.csv")
    falcon_data = parse_csv(RESULTS_DIR / "F5_full_metrics.csv")

    # Extract epoch-level data
    adamw_epochs = get_epoch_data(adamw_data)
    muon_epochs = get_epoch_data(muon_data)
    falcon_epochs = get_epoch_data(falcon_data)

    # Build the JSON structure
    dynamics = {
        "trainingLoss": {
            "epochs": [e['epoch'] for e in falcon_epochs],
            "AdamW": [e['train_loss'] for e in adamw_epochs],
            "Muon": [e['train_loss'] for e in muon_epochs],
            "Falcon": [e['train_loss'] for e in falcon_epochs]
        },
        "validationAccuracy": {
            "epochs": [e['epoch'] for e in falcon_epochs],
            "AdamW": [e['val_acc'] / 100 if e['val_acc'] else None for e in adamw_epochs],
            "Muon": [e['val_acc'] / 100 if e['val_acc'] else None for e in muon_epochs],
            "Falcon": [e['val_acc'] / 100 if e['val_acc'] else None for e in falcon_epochs]
        },
        "retainFraction": {
            "epochs": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 236],
            "values": [0.95, 0.905, 0.86, 0.815, 0.77, 0.725, 0.68, 0.635, 0.59, 0.545, 0.5]
        },
        "interleavingPeriod": {
            "epochs": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 236],
            "values": [4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1]
        },
        "epochTimes": {
            "AdamW": 4.8,
            "Muon": 5.3,
            "Falcon": 6.7
        },
        "finalAccuracies": {
            "AdamW": 90.28,
            "Muon": 90.49,
            "Falcon": 90.33
        },
        "throughput": {
            "AdamW": 10382,
            "Muon": 9418,
            "Falcon": 7486
        },
        "convergenceTime": {
            "AdamW": 1.27,
            "Muon": 1.18,
            "Falcon": 1.56
        }
    }

    # Save to file
    output_path = OUTPUT_DIR / "dynamics.json"
    with open(output_path, 'w') as f:
        json.dump(dynamics, f, indent=2)

    print(f"✓ Created {output_path}")
    print(f"  - {len(falcon_epochs)} epochs of training data")
    print(f"  - AdamW final accuracy: {dynamics['finalAccuracies']['AdamW']}%")
    print(f"  - Muon final accuracy: {dynamics['finalAccuracies']['Muon']}%")
    print(f"  - Falcon final accuracy: {dynamics['finalAccuracies']['Falcon']}%")

def create_summary_json():
    """Create a summary JSON from table_summary.csv"""

    summary_data = parse_csv(RESULTS_DIR / "table_summary.csv")

    # Organize by experiment type
    summary = {
        "full_training": [],
        "fixed_time": [],
        "data_efficiency": []
    }

    for row in summary_data:
        exp_type = row['Type']
        entry = {
            "experiment": row['Experiment'],
            "optimizer": row['Optimizer'],
            "data_fraction": float(row['Data Fraction']),
            "best_val_acc": float(row['Best Val@1 (%)']),
            "best_epoch": int(row['Best Epoch']),
            "total_time_min": float(row['Total Time (min)']),
            "median_epoch_time_s": float(row['Median Epoch Time (s)']),
            "images_per_sec": int(row['Images/sec']),
            "time_to_85_percent": float(row['Time to 85% (min)']) if row['Time to 85% (min)'] != '0.00' else None
        }

        if exp_type == 'full':
            summary["full_training"].append(entry)
        elif exp_type == 'fixed_time':
            summary["fixed_time"].append(entry)
        elif exp_type == 'data_eff':
            summary["data_efficiency"].append(entry)

    # Save to file
    output_path = OUTPUT_DIR / "summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Created {output_path}")

if __name__ == "__main__":
    print("Converting Falcon v5 CSV data to JSON...")
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create the JSON files
    create_dynamics_json()
    print()
    create_summary_json()
    print()
    print("✓ Data conversion complete!")
