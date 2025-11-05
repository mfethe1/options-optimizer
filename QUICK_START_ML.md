# 🚀 Quick Start: ML Environment Setup

## TL;DR - Get ML Features Working in 2 Minutes

```bash
# 1. Run the automated setup
./setup_ml_env.sh

# 2. Activate the environment
source venv_ml/bin/activate

# 3. Start the server
uvicorn src.main:app --reload

# 4. Test ML endpoints
curl http://localhost:8000/api/advanced-forecast/status
```

## Why You Need This

The Options Optimizer uses cutting-edge neural network models:
- **TFT (Temporal Fusion Transformer)** - 11% better than LSTM
- **GNN (Graph Neural Networks)** - 20-30% improvement via correlations
- **PINN (Physics-Informed Neural Networks)** - 15-100x data efficiency
- **Mamba State-Space Models** - Long-range dependencies
- **Epidemic Volatility Models** - Bio-financial crossover

## Python 3.12 Issue ⚠️

**TensorFlow 2.15 does NOT support Python 3.12!**

We've upgraded to **TensorFlow 2.16+** which supports Python 3.9-3.12.

## Supported Python Versions

| Version | Status      | Notes                    |
|---------|-------------|--------------------------|
| 3.11    | ✅ Best     | Recommended              |
| 3.12    | ✅ Supported| Requires TensorFlow 2.16+|
| 3.10    | ✅ Supported| Stable                   |
| 3.9     | ✅ Supported| Older but works          |

## Setup Options

### Option 1: Automated (Recommended)
```bash
./setup_ml_env.sh
```

### Option 2: Manual
```bash
python3.11 -m venv venv_ml
source venv_ml/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Verification

```bash
source venv_ml/bin/activate
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} ready!')"
```

## Full Documentation

See [ML_ENVIRONMENT_SETUP.md](ML_ENVIRONMENT_SETUP.md) for complete details.

## Troubleshooting

**Problem**: `name 'layers' is not defined`  
**Solution**: ✅ Already fixed! The code now gracefully handles missing TensorFlow.

**Problem**: Python 3.12 compatibility  
**Solution**: ✅ Already fixed! Updated to TensorFlow 2.16+

**Problem**: Virtual environment issues  
**Solution**: Delete `venv_ml/` and run `./setup_ml_env.sh` again

## Next Steps

1. ✅ Run `./setup_ml_env.sh`
2. ✅ Activate with `source venv_ml/bin/activate`
3. ✅ Start server with `uvicorn src.main:app --reload`
4. ✅ Test at http://localhost:8000/docs
