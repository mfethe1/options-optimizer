# ML Environment Setup Guide

This guide explains how to set up the Python environment for the neural network and machine learning components of the Options Optimizer system.

## 🎯 Overview

The system uses advanced ML models that require TensorFlow:
- **Temporal Fusion Transformer (TFT)** - Multi-horizon forecasting
- **Graph Neural Networks (GNN)** - Stock correlation modeling
- **Physics-Informed Neural Networks (PINN)** - Option pricing with PDE constraints
- **Mamba State-Space Models** - Long-range dependency modeling
- **Epidemic Volatility Models** - Bio-financial volatility forecasting

## 📋 Python Version Compatibility

| Python Version | TensorFlow 2.15 | TensorFlow 2.16+ | Recommended |
|----------------|-----------------|------------------|-------------|
| 3.9            | ✅ Supported    | ✅ Supported     | ✅          |
| 3.10           | ✅ Supported    | ✅ Supported     | ✅          |
| 3.11           | ✅ Supported    | ✅ Supported     | ✅ **Best** |
| 3.12           | ❌ Not supported | ✅ Supported     | ✅          |

**Recommendation**: Use **Python 3.11** for best stability and performance.

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
chmod +x setup_ml_env.sh
./setup_ml_env.sh

# Activate the environment
source venv_ml/bin/activate
```

### Option 2: Manual Setup

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv_ml

# Activate virtual environment
source venv_ml/bin/activate

# Upgrade core tools
pip install --upgrade pip setuptools wheel

# Install TensorFlow 2.16+ (supports Python 3.9-3.12)
pip install tensorflow>=2.16.0

# Install all project dependencies
pip install -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

## 🔧 Detailed Setup Steps

### Step 1: Check Python Version

```bash
# Check available Python versions
python3.11 --version  # Recommended
python3.12 --version  # Also works with TensorFlow 2.16+
python3.10 --version  # Also works
```

### Step 2: Create Virtual Environment

```bash
# Using Python 3.11 (recommended)
python3.11 -m venv venv_ml

# Or using Python 3.12 (requires TensorFlow 2.16+)
python3.12 -m venv venv_ml
```

### Step 3: Activate Virtual Environment

```bash
# Linux/Mac
source venv_ml/bin/activate

# Windows
venv_ml\Scripts\activate
```

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install TensorFlow 2.16+ (Python 3.9-3.12 compatible)
pip install tensorflow>=2.16.0

# Install all project requirements
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Verify TensorFlow
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Verify GPU support (if applicable)
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"

# Test import of ML modules
python -c "from src.ml.advanced_forecasting.tft_model import TemporalFusionTransformer; print('✓ TFT model OK')"
python -c "from src.ml.graph_neural_network.stock_gnn import StockGNN; print('✓ GNN model OK')"
python -c "from src.ml.physics_informed.general_pinn import OptionPricingPINN; print('✓ PINN model OK')"
```

## 🐛 Troubleshooting

### Issue: Python 3.12 with TensorFlow 2.15

**Error**: `tensorflow 2.15.0 requires Python >=3.9, <3.12`

**Solution**: Either:
1. Upgrade to TensorFlow 2.16+: `pip install --upgrade tensorflow>=2.16.0`
2. Use Python 3.11: Create venv with `python3.11 -m venv venv_ml`

### Issue: Import errors for `layers` or `tf`

**Error**: `NameError: name 'layers' is not defined`

**Solution**: This is now handled gracefully. The system will:
- Use TensorFlow features if available
- Fall back to simplified implementations if TensorFlow is not installed
- Log warnings about missing TensorFlow functionality

### Issue: M1/M2 Mac ARM architecture

For Apple Silicon Macs:

```bash
# Use tensorflow-macos and tensorflow-metal for GPU acceleration
pip install tensorflow-macos>=2.16.0
pip install tensorflow-metal>=1.1.0
```

### Issue: CUDA/GPU support on Linux

For NVIDIA GPU support:

```bash
# Check CUDA version
nvidia-smi

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]>=2.16.0
```

## 📦 What Gets Installed

### Core ML Libraries
- **TensorFlow 2.16+** - Neural network framework
- **NumPy 1.26+** - Numerical computing
- **SciPy 1.11+** - Scientific computing
- **Pandas 2.1+** - Data manipulation
- **scikit-learn 1.3+** - Classical ML algorithms
- **statsmodels 0.14+** - Statistical modeling

### TensorFlow Components Used
- `tensorflow.keras.layers` - Neural network layers
- `tensorflow.keras.Model` - Model base class
- `tensorflow.GradientTape` - Automatic differentiation (for PINNs)
- `tensorflow.keras.optimizers` - Training optimizers

## 🎮 Usage

### Activating the Environment

Every time you work on ML components:

```bash
# Quick activation
source venv_ml/bin/activate

# Or use the helper script
./activate_ml_env.sh
```

### Running ML Endpoints

```bash
# Activate environment first
source venv_ml/bin/activate

# Start the server
uvicorn src.main:app --reload

# Test ML endpoints
curl http://localhost:8000/api/advanced-forecast/status
curl http://localhost:8000/api/gnn/status
curl http://localhost:8000/api/pinn/status
```

### Training Models

```bash
# Activate environment
source venv_ml/bin/activate

# Train TFT model
curl -X POST http://localhost:8000/api/advanced-forecast/train \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"], "epochs": 50}'

# Train PINN model
curl -X POST http://localhost:8000/api/pinn/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "options", "epochs": 1000}'
```

### Deactivating the Environment

```bash
deactivate
```

## 🔍 Verifying Your Setup

Run this comprehensive test:

```bash
# Activate environment
source venv_ml/bin/activate

# Create test script
cat > test_ml_setup.py << 'EOF'
import sys
print(f"Python version: {sys.version}")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

import numpy as np
import pandas as pd
import sklearn
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")

# Test ML module imports
from src.ml.advanced_forecasting.tft_model import TemporalFusionTransformer
from src.ml.graph_neural_network.stock_gnn import StockGNN
from src.ml.physics_informed.general_pinn import OptionPricingPINN

print("\n✅ All ML components verified successfully!")
EOF

# Run test
python test_ml_setup.py

# Clean up
rm test_ml_setup.py
```

## 📚 Additional Resources

- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [TensorFlow 2.16 Release Notes](https://github.com/tensorflow/tensorflow/releases/tag/v2.16.0)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

## 🔄 Updating Dependencies

To update TensorFlow and ML dependencies:

```bash
# Activate environment
source venv_ml/bin/activate

# Update TensorFlow
pip install --upgrade tensorflow

# Update all ML dependencies
pip install --upgrade numpy scipy pandas scikit-learn statsmodels

# Or update all requirements
pip install --upgrade -r requirements.txt
```

## 🗑️ Removing the Environment

If you need to start fresh:

```bash
# Deactivate first (if active)
deactivate

# Remove the virtual environment
rm -rf venv_ml

# Run setup again
./setup_ml_env.sh
```

## 💡 Pro Tips

1. **Always activate the environment** before working on ML features
2. **Use Python 3.11** for best compatibility and performance
3. **Check GPU availability** if you have an NVIDIA GPU for faster training
4. **Keep TensorFlow updated** for latest features and bug fixes
5. **Use the helper script** `activate_ml_env.sh` for quick activation

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify your Python version is 3.9-3.12
3. Ensure TensorFlow 2.16+ is installed
4. Check the logs for specific error messages
5. Verify all imports work in the test script above
