#!/bin/bash

echo "Running TP1 smoke tests..."

# Test Python environment
echo "Testing Python installation..."
python --version
if [ $? -ne 0 ]; then
    echo "Error: Python not found"
    exit 1
fi

# Test if TensorFlow is installed
echo "Testing TensorFlow installation..."
python -c "import tensorflow as tf; print(tf.__version__)"
if [ $? -ne 0 ]; then
    echo "Error: TensorFlow not installed"
    exit 1
fi

# Test if the training script exists and is readable
echo "Testing training script..."
if [ ! -f "train_model.py" ]; then
    echo "Error: train_model.py not found"
    exit 1
fi

echo "All smoke tests passed!"