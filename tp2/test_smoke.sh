#!/bin/bash

# Test script for basic functionality

echo "Running smoke tests..."

# Test Python environment
echo "Testing Python installation..."
python --version
if [ $? -ne 0 ]; then
    echo "Error: Python not found"
    exit 1
fi

# Test dry run of training script
echo "Testing training script (dry run)..."
python train_model.py --dry-run
if [ $? -ne 0 ]; then
    echo "Error: Training script failed"
    exit 1
fi

echo "All smoke tests passed!"