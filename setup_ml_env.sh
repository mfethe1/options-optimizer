#!/bin/bash
#
# ML Environment Setup Script
# Sets up Python virtual environment with TensorFlow and neural network dependencies
#
# Supports: Python 3.9, 3.10, 3.11, 3.12
# TensorFlow 2.16+ has full Python 3.12 support
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}   ML Environment Setup${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Detect Python version
echo -e "${YELLOW}Detecting Python version...${NC}"

# Try to find the best Python version
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${GREEN}✓ Found Python 3.11 (recommended for TensorFlow)${NC}"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}✓ Found Python 3.10 (compatible with TensorFlow)${NC}"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo -e "${GREEN}✓ Found Python 3.12 (requires TensorFlow 2.16+)${NC}"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo -e "${GREEN}✓ Found Python 3.9 (compatible with TensorFlow)${NC}"
else
    echo -e "${RED}✗ No compatible Python version found (need 3.9-3.12)${NC}"
    exit 1
fi

# Display Python version
PYTHON_VERSION=$($PYTHON_CMD --version)
echo -e "Using: ${GREEN}${PYTHON_VERSION}${NC}"
echo ""

# Check if venv already exists
VENV_DIR="venv_ml"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at ${VENV_DIR}${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${GREEN}Using existing virtual environment${NC}"
        source "${VENV_DIR}/bin/activate"
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
        exit 0
    fi
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv "$VENV_DIR"
echo -e "${GREEN}✓ Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "${VENV_DIR}/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip, setuptools, wheel
echo -e "${YELLOW}Upgrading pip, setuptools, and wheel...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Core tools upgraded${NC}"
echo ""

# Install TensorFlow and ML dependencies
echo -e "${YELLOW}Installing TensorFlow and ML dependencies...${NC}"
echo -e "${BLUE}This may take several minutes...${NC}"
echo ""

# Install TensorFlow 2.16+ (supports Python 3.9-3.12)
pip install tensorflow>=2.16.0

# Install other ML libraries
pip install numpy>=1.26.0
pip install scipy>=1.11.0
pip install pandas>=2.1.0
pip install scikit-learn>=1.3.0
pip install statsmodels>=0.14.0

echo -e "${GREEN}✓ ML dependencies installed${NC}"
echo ""

# Install the main project requirements
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Installing main project requirements...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Project requirements installed${NC}"
    echo ""
fi

# Verify TensorFlow installation
echo -e "${YELLOW}Verifying TensorFlow installation...${NC}"
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
echo -e "${GREEN}✓ TensorFlow verified${NC}"
echo ""

# Create activation helper script
echo -e "${YELLOW}Creating activation helper script...${NC}"
cat > activate_ml_env.sh << 'EOF'
#!/bin/bash
# Quick activation script for ML environment
source venv_ml/bin/activate
echo "ML environment activated!"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo ""
echo "To deactivate, run: deactivate"
EOF
chmod +x activate_ml_env.sh
echo -e "${GREEN}✓ Created activate_ml_env.sh${NC}"
echo ""

# Summary
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}   Setup Complete! 🎉${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo -e "To activate the ML environment in the future:"
echo -e "  ${BLUE}source venv_ml/bin/activate${NC}"
echo -e "  or"
echo -e "  ${BLUE}./activate_ml_env.sh${NC}"
echo ""
echo -e "To deactivate:"
echo -e "  ${BLUE}deactivate${NC}"
echo ""
echo -e "Python version: ${GREEN}$PYTHON_VERSION${NC}"
echo -e "TensorFlow location: ${GREEN}venv_ml/lib/python*/site-packages/tensorflow${NC}"
echo ""
