#!/usr/bin/env python3
"""
Email notification utility for FALCON v5 experiments.
Sends email when training completes (success or failure).

Usage:
    python scripts/notify.py --email your.email@example.com --status success --message "Training complete!"
    python scripts/notify.py --email your.email@example.com --status failure --message "Training crashed at epoch 30"
"""

import argparse
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_email(to_email, subject, body, status="success"):
    """
    Send email notification using system mail command (simple method).
    Works on Linux systems with mail/sendmail configured.
    """
    try:
        # Method 1: Use system 'mail' command (simplest, works on most Linux)
        import subprocess
        
        message = f"""
FALCON v5 Experiment Notification
==================================

Status: {status.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Host: {socket.gethostname()}

Message:
{body}

==================================
This is an automated notification from your FALCON v5 training script.
"""
        
        # Try using mail command
        try:
            subprocess.run(
                ['mail', '-s', subject, to_email],
                input=message.encode(),
                check=True,
                timeout=10
            )
            print(f"✓ Email sent to {to_email} via 'mail' command")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Method 2: Try sendmail
        try:
            sendmail_path = '/usr/sbin/sendmail'
            email_text = f"""From: FALCON v5 <noreply@{socket.gethostname()}>
To: {to_email}
Subject: {subject}

{message}
"""
            subprocess.run(
                [sendmail_path, to_email],
                input=email_text.encode(),
                check=True,
                timeout=10
            )
            print(f"✓ Email sent to {to_email} via 'sendmail'")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Method 3: Print to console if email fails
        print("\n" + "="*60)
        print("⚠️  EMAIL NOTIFICATION (delivery failed, printed here):")
        print("="*60)
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(message)
        print("="*60)
        print("\nTo enable email delivery, configure 'mail' or 'sendmail' on your system.")
        print("Or use --smtp-server option (see --help)\n")
        return False
        
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        print("Notification message (printed):")
        print(body)
        return False


def send_email_smtp(to_email, subject, body, smtp_server, smtp_port, username, password, status="success"):
    """
    Send email via SMTP server (more reliable, requires credentials).
    
    Example SMTP servers:
    - Gmail: smtp.gmail.com:587 (requires app password)
    - Outlook: smtp-mail.outlook.com:587
    - Office365: smtp.office365.com:587
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Email body
        email_body = f"""
<html>
<body>
<h2>FALCON v5 Experiment Notification</h2>
<table border="1" cellpadding="5">
    <tr><td><b>Status</b></td><td style="color: {'green' if status == 'success' else 'red'};">{status.upper()}</td></tr>
    <tr><td><b>Time</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
    <tr><td><b>Host</b></td><td>{socket.gethostname()}</td></tr>
</table>
<br>
<h3>Message:</h3>
<pre>{body}</pre>
<br>
<hr>
<p><i>This is an automated notification from your FALCON v5 training script.</i></p>
</body>
</html>
"""
        msg.attach(MIMEText(email_body, 'html'))
        
        # Connect to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure connection
        server.login(username, password)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email sent to {to_email} via SMTP ({smtp_server})")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send email via SMTP: {e}")
        print("Notification message (printed):")
        print(body)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Send email notification for FALCON v5 experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple (uses system mail):
  python scripts/notify.py --email noel.thomas@example.com --status success --message "All experiments complete!"
  
  # With SMTP (Gmail example):
  python scripts/notify.py --email noel.thomas@example.com --status success --message "Training done!" \\
    --smtp-server smtp.gmail.com --smtp-port 587 --smtp-user your.email@gmail.com --smtp-password "your_app_password"
  
  # Failure notification:
  python scripts/notify.py --email noel.thomas@example.com --status failure --message "Training crashed at epoch 45"
"""
    )
    
    parser.add_argument('--email', required=True, help='Email address to send notification to')
    parser.add_argument('--status', choices=['success', 'failure', 'info'], default='success',
                        help='Notification status (default: success)')
    parser.add_argument('--message', required=True, help='Notification message body')
    parser.add_argument('--subject', help='Email subject (auto-generated if not provided)')
    
    # SMTP options (optional, for more reliable delivery)
    parser.add_argument('--smtp-server', help='SMTP server (e.g., smtp.gmail.com)')
    parser.add_argument('--smtp-port', type=int, default=587, help='SMTP port (default: 587)')
    parser.add_argument('--smtp-user', help='SMTP username (email address)')
    parser.add_argument('--smtp-password', help='SMTP password or app password')
    
    args = parser.parse_args()
    
    # Auto-generate subject if not provided
    if args.subject is None:
        emoji = "✅" if args.status == "success" else "❌" if args.status == "failure" else "ℹ️"
        args.subject = f"{emoji} FALCON v5: {args.status.capitalize()}"
    
    # Send via SMTP if credentials provided
    if args.smtp_server and args.smtp_user and args.smtp_password:
        success = send_email_smtp(
            args.email, args.subject, args.message,
            args.smtp_server, args.smtp_port,
            args.smtp_user, args.smtp_password,
            args.status
        )
    else:
        # Fall back to system mail
        success = send_email(args.email, args.subject, args.message, args.status)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
