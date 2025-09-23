#!/bin/bash

# Automated Installation Script for Chest X-ray AI POC
# For use on Ubuntu 22.04 with NVIDIA GPU support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Starting Chest X-ray AI POC installation..."

# Update system
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install Python and essential tools
print_status "Installing Python and essential tools..."
apt install -y python3 python3-pip python3-dev wget curl git vim htop tree

# Install system dependencies for medical imaging
print_status "Installing system dependencies..."
apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    
    # Install PyTorch with CUDA support
    print_status "Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_warning "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install project dependencies
print_status "Installing project dependencies..."
pip3 install -r requirements.txt

# Pre-download AI models
print_status "Downloading AI models (this may take a few minutes)..."
python3 -c "
import torchxrayvision as xrv
print('Downloading DenseNet121 model...')
try:
    model = xrv.models.get_model('densenet121-res224-all')
    print('✓ Model downloaded successfully')
except Exception as e:
    print(f'✗ Model download failed: {e}')
    exit(1)
"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs uploads models

# Install process manager (PM2)
print_status "Installing PM2 process manager..."
if command -v npm &> /dev/null; then
    npm install -g pm2
    print_success "PM2 installed"
else
    print_warning "npm not found. Installing Node.js and PM2..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt install -y nodejs
    npm install -g pm2
fi

# Create PM2 ecosystem file
print_status "Creating PM2 configuration..."
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'xray-api',
      script: 'python3',
      args: ['-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'],
      cwd: process.cwd(),
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      env: {
        NODE_ENV: 'production'
      }
    },
    {
      name: 'xray-frontend',
      script: 'python3',
      args: ['-m', 'http.server', '3000'],
      cwd: process.cwd() + '/frontend',
      instances: 1,
      autorestart: true,
      watch: false,
    }
  ]
};
EOF

# Setup firewall
print_status "Configuring firewall..."
ufw allow ssh
ufw allow 80
ufw allow 3000
ufw allow 8000
ufw --force enable

# Test the installation
print_status "Testing the installation..."
python3 -c "
import torch
import torchxrayvision as xrv
from fastapi import FastAPI

print('✓ PyTorch installed')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU count: {torch.cuda.device_count()}')
    print(f'✓ GPU name: {torch.cuda.get_device_name(0)}')
print('✓ TorchXRayVision installed')
print('✓ FastAPI installed')
print('✓ All dependencies working')
"

print_success "Installation completed successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Start the application:"
echo "   pm2 start ecosystem.config.js"
echo ""
echo "2. Access your application:"
echo "   Frontend: http://$(curl -s ifconfig.me):3000"
echo "   API Docs: http://$(curl -s ifconfig.me):8000/docs"
echo ""
echo "3. Monitor the application:"
echo "   pm2 status"
echo "   pm2 logs"
echo ""
echo "4. Stop the application:"
echo "   pm2 stop all"
echo ""
print_success "Happy analyzing! 🩻"
