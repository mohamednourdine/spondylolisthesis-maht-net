#!/usr/bin/env python3
"""Simple wrapper to run training with immediate output."""

import subprocess
import sys

cmd = [
    sys.executable,
    "train.py",
    "--model", "unet",
    "--epochs", "1", 
    "--batch-size", "2",
    "--experiment-name", "test_metrics"
]

print("Starting training...")
print(f"Command: {' '.join(cmd)}\n")

# Run with unbuffered output
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

try:
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    print(f"\nProcess exited with code: {process.returncode}")
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    process.terminate()
    sys.exit(1)
