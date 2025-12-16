#!/usr/bin/env python3
"""Minimal test of training entry point."""

import sys
import subprocess

print("Testing train.py --help...")
result = subprocess.run(
    ["python", "train.py", "--help"],
    capture_output=True,
    text=True,
    cwd="/Users/mnourdine/phd/spondylolisthesis-maht-net"
)

print(result.stdout)
if result.returncode != 0:
    print(f"Error: {result.stderr}")
    sys.exit(1)

print("\n✓ train.py --help works!")
