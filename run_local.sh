#!/bin/bash

# Local Development Script for Chest X-ray AI POC
# This script sets up and runs the application locally

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        print_status "Checking GPU availability..."
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        print_success "GPU detected"
    else
        print_warning "nvidia-smi not found. Running on CPU (will be slower)"
    fi
}

# Function to create virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install PyTorch with appropriate CUDA version
    if command_exists nvidia-smi; then
        print_status "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch CPU-only version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other requirements
    print_status "Installing other requirements..."
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Function to download models
download_models() {
    print_status "Downloading AI models (this may take a few minutes)..."
    python3 -c "
import torchxrayvision as xrv
print('Downloading models...')
try:
    model = xrv.models.get_model('densenet121-res224-all')
    print('✓ DenseNet121 model downloaded')
except Exception as e:
    print(f'✗ Model download failed: {e}')
    exit(1)
"
    print_success "Models downloaded successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    if command_exists pytest; then
        pytest tests/ -v
        print_success "Tests completed"
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Function to start the application
start_application() {
    print_status "Starting the application..."
    
    # Create necessary directories
    mkdir -p logs uploads models
    
    # Start API server in background
    print_status "Starting API server on http://localhost:8000..."
    python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for API to start
    sleep 5
    
    # Check if API is running
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "API server started successfully"
    else
        print_error "API server failed to start"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    
    # Start frontend server
    print_status "Starting frontend server on http://localhost:3000..."
    cd frontend
    python3 -m http.server 3000 &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    sleep 2
    
    print_success "Application started successfully!"
    echo ""
    echo "🚀 Access your application:"
    echo "   Frontend: http://localhost:3000"
    echo "   API Docs: http://localhost:8000/docs"
    echo "   Health Check: http://localhost:8000/health"
    echo ""
    print_status "Press Ctrl+C to stop the application"
    
    # Function to cleanup on exit
    cleanup() {
        print_status "Shutting down servers..."
        kill $API_PID 2>/dev/null || true
        kill $FRONTEND_PID 2>/dev/null || true
        print_success "Application stopped"
        exit 0
    }
    
    # Set trap for cleanup
    trap cleanup INT TERM
    
    # Wait for user to stop
    wait
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup     - Setup environment and install dependencies"
    echo "  run       - Run the application (setup first if needed)"
    echo "  test      - Run tests"
    echo "  clean     - Clean up virtual environment"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # First time setup"
    echo "  $0 run       # Run the application"
    echo "  $0 test      # Run tests"
}

# Main script logic
main() {
    case "${1:-run}" in
        "setup")
            print_status "Setting up Chest X-ray AI POC..."
            check_python || exit 1
            check_gpu
            setup_venv
            install_dependencies
            download_models
            print_success "Setup completed successfully!"
            echo ""
            echo "Next steps:"
            echo "  1. Run: $0 run"
            echo "  2. Open http://localhost:3000 in your browser"
            echo "  3. Upload a chest X-ray image to test"
            ;;
        
        "run")
            print_status "Starting Chest X-ray AI POC..."
            
            # Check if setup is needed
            if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
                print_warning "Virtual environment not found. Running setup first..."
                main setup
            fi
            
            # Activate virtual environment
            source venv/bin/activate
            
            # Check if dependencies are installed
            if ! python3 -c "import torchxrayvision" >/dev/null 2>&1; then
                print_warning "Dependencies not found. Installing..."
                install_dependencies
                download_models
            fi
            
            start_application
            ;;
        
        "test")
            print_status "Running tests..."
            if [ ! -d "venv" ]; then
                print_error "Virtual environment not found. Run setup first."
                exit 1
            fi
            source venv/bin/activate
            run_tests
            ;;
        
        "clean")
            print_status "Cleaning up..."
            if [ -d "venv" ]; then
                rm -rf venv
                print_success "Virtual environment removed"
            fi
            if [ -d "__pycache__" ]; then
                find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
                print_success "Cache files removed"
            fi
            ;;
        
        "help"|"-h"|"--help")
            show_usage
            ;;
        
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
