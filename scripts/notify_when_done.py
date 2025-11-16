#!/usr/bin/env python3
"""
Monitor training runs and send email notification when complete.

Usage:
    python scripts/notify_when_done.py --email your.email@example.com --exp F5_full
    
Or monitor all experiments:
    python scripts/notify_when_done.py --email your.email@example.com --all
"""

import argparse
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import subprocess

def send_email(to_email, subject, body, from_email="falcon.notify@automail.com"):
    """
    Send email notification using system mail command.
    Falls back to different methods if mail command not available.
    """
    # Method 1: Try using system 'mail' command (most reliable on Linux)
    try:
        mail_cmd = f'echo "{body}" | mail -s "{subject}" {to_email}'
        result = subprocess.run(mail_cmd, shell=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ Email sent to {to_email} via system mail")
            return True
    except Exception as e:
        print(f"System mail failed: {e}")
    
    # Method 2: Try using 'sendmail' command
    try:
        msg = f"Subject: {subject}\n\n{body}"
        sendmail_cmd = f'echo "{msg}" | /usr/sbin/sendmail {to_email}'
        result = subprocess.run(sendmail_cmd, shell=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ Email sent to {to_email} via sendmail")
            return True
    except Exception as e:
        print(f"Sendmail failed: {e}")
    
    # Method 3: Print notification (fallback)
    print(f"\n{'='*60}")
    print(f"EMAIL NOTIFICATION (mail command not available)")
    print(f"To: {to_email}")
    print(f"Subject: {subject}")
    print(f"\n{body}")
    print(f"{'='*60}\n")
    
    # Also write to a notification file
    notify_file = Path("TRAINING_COMPLETE_NOTIFICATION.txt")
    with open(notify_file, "w") as f:
        f.write(f"To: {to_email}\n")
        f.write(f"Subject: {subject}\n\n")
        f.write(body)
    print(f"✓ Notification saved to {notify_file}")
    
    return False

def check_experiment_complete(exp_name, runs_dir="runs"):
    """
    Check if an experiment is complete by looking for:
    1. best.pt checkpoint exists
    2. metrics.csv has multiple entries
    3. No recent updates (file not being written to)
    """
    exp_dir = Path(runs_dir) / exp_name
    
    if not exp_dir.exists():
        return False, "Experiment directory not found"
    
    best_pt = exp_dir / "best.pt"
    metrics_csv = exp_dir / "metrics.csv"
    
    if not best_pt.exists():
        return False, "No checkpoint saved yet"
    
    if not metrics_csv.exists():
        return False, "No metrics logged yet"
    
    # Check if metrics file has been recently modified (within last 60 seconds)
    try:
        mtime = metrics_csv.stat().st_mtime
        age = time.time() - mtime
        if age < 60:
            return False, f"Still training (last update {age:.0f}s ago)"
    except Exception as e:
        return False, f"Error checking file: {e}"
    
    # Check metrics file has multiple epochs
    try:
        with open(metrics_csv, 'r') as f:
            lines = f.readlines()
            if len(lines) < 3:  # Header + at least 2 epochs
                return False, "Training just started"
    except Exception as e:
        return False, f"Error reading metrics: {e}"
    
    return True, "Complete"

def get_experiment_summary(exp_name, runs_dir="runs"):
    """Extract key metrics from experiment."""
    exp_dir = Path(runs_dir) / exp_name
    metrics_csv = exp_dir / "metrics.csv"
    
    try:
        with open(metrics_csv, 'r') as f:
            lines = f.readlines()
        
        # Parse last line for final metrics
        last_line = lines[-1].strip().split(',')
        
        # Parse all lines for best accuracy
        best_val = 0.0
        best_epoch = 0
        total_epochs = 0
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 5:
                epoch = int(parts[0])
                total_epochs = max(total_epochs, epoch)
                
                val_acc_str = parts[4]
                if val_acc_str:
                    val_acc = float(val_acc_str)
                    if val_acc > best_val:
                        best_val = val_acc
                        best_epoch = epoch
        
        wall_min = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        
        return {
            'best_val': best_val,
            'best_epoch': best_epoch,
            'total_epochs': total_epochs,
            'wall_min': wall_min,
        }
    except Exception as e:
        return {'error': str(e)}

def monitor_experiments(email, experiments, check_interval=60, runs_dir="runs"):
    """
    Monitor multiple experiments and send notification when all complete.
    
    Args:
        email: Email address to notify
        experiments: List of experiment names to monitor
        check_interval: Seconds between checks (default 60)
        runs_dir: Directory containing experiment runs
    """
    print(f"🔍 Monitoring {len(experiments)} experiment(s)...")
    print(f"📧 Will notify: {email}")
    print(f"⏱️  Check interval: {check_interval}s")
    print(f"📁 Runs directory: {runs_dir}")
    print()
    
    completed = set()
    start_time = time.time()
    
    while True:
        for exp_name in experiments:
            if exp_name in completed:
                continue
            
            is_complete, status = check_experiment_complete(exp_name, runs_dir)
            
            if is_complete:
                print(f"✅ {exp_name}: {status}")
                completed.add(exp_name)
                
                # Get summary
                summary = get_experiment_summary(exp_name, runs_dir)
                
                # Send individual notification
                subject = f"FALCON Training Complete: {exp_name}"
                body = f"""
Experiment: {exp_name}
Status: Complete ✓

Results:
- Best Validation Accuracy: {summary.get('best_val', 'N/A'):.2f}%
- Best Epoch: {summary.get('best_epoch', 'N/A')}
- Total Epochs: {summary.get('total_epochs', 'N/A')}
- Total Time: {summary.get('wall_min', 0):.1f} minutes

Checkpoint saved: runs/{exp_name}/best.pt
Metrics: runs/{exp_name}/metrics.csv

Time to completion: {(time.time() - start_time) / 60:.1f} minutes
"""
                send_email(email, subject, body)
            else:
                print(f"⏳ {exp_name}: {status}")
        
        # Check if all complete
        if len(completed) == len(experiments):
            print(f"\n🎉 All {len(experiments)} experiment(s) complete!")
            
            # Send final summary email
            subject = f"FALCON Training Suite Complete ({len(experiments)} experiments)"
            body = "All experiments completed!\n\n"
            
            for exp_name in experiments:
                summary = get_experiment_summary(exp_name, runs_dir)
                body += f"{exp_name}:\n"
                body += f"  Best Val@1: {summary.get('best_val', 'N/A'):.2f}%\n"
                body += f"  Time: {summary.get('wall_min', 0):.1f} min\n\n"
            
            body += f"\nTotal monitoring time: {(time.time() - start_time) / 60:.1f} minutes\n"
            body += f"\nNext steps:\n"
            body += f"  python scripts/plot_results_v5.py\n"
            body += f"  Check paper_assets/ for figures\n"
            
            send_email(email, subject, body)
            break
        
        # Wait before next check
        print(f"⏸️  Waiting {check_interval}s before next check...\n")
        time.sleep(check_interval)

def main():
    parser = argparse.ArgumentParser(
        description="Monitor FALCON training and send email notification when complete"
    )
    parser.add_argument(
        '--email', '-e',
        required=True,
        help='Email address to send notification'
    )
    parser.add_argument(
        '--exp', '-x',
        nargs='+',
        help='Experiment name(s) to monitor (e.g., F5_full A1_full)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Monitor all experiments in runs/ directory'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Check interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--runs-dir',
        default='runs',
        help='Directory containing experiment runs (default: runs)'
    )
    
    args = parser.parse_args()
    
    # Determine which experiments to monitor
    if args.all:
        runs_path = Path(args.runs_dir)
        if not runs_path.exists():
            print(f"❌ Error: {args.runs_dir} directory not found")
            return
        
        experiments = [d.name for d in runs_path.iterdir() if d.is_dir()]
        experiments = [e for e in experiments if not e.startswith('.')]
        
        if not experiments:
            print(f"❌ No experiments found in {args.runs_dir}/")
            return
        
        print(f"📋 Found {len(experiments)} experiment(s):")
        for exp in experiments:
            print(f"   - {exp}")
        print()
    elif args.exp:
        experiments = args.exp
    else:
        print("❌ Error: Specify --exp EXPERIMENT or --all")
        parser.print_help()
        return
    
    # Validate email format (basic check)
    if '@' not in args.email:
        print(f"⚠️  Warning: '{args.email}' doesn't look like a valid email")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Start monitoring
    try:
        monitor_experiments(args.email, experiments, args.interval, args.runs_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        print(f"Completed: {len([e for e in experiments if check_experiment_complete(e, args.runs_dir)[0]])}/{len(experiments)}")

if __name__ == '__main__':
    main()
