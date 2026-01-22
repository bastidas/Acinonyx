#!/bin/bash

# Acinonyx Project Launch Script
# Author: Acinonyx Team
# Date: November 15, 2025

set -e  # Exit on any error

# Parse arguments
SKIP_INSTALL=true
for arg in "$@"; do
    case $arg in
        -q|--quick)
            SKIP_INSTALL=true
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - Get from appconfig.py
BACKEND_PORT=$(python -c "from configs.appconfig import BACKEND_PORT; print(BACKEND_PORT)")
FRONTEND_PORT=$(python -c "from configs.appconfig import FRONTEND_PORT; print(FRONTEND_PORT)")
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"

# Print banner
echo -e "${BLUE}"
echo "    🐆 ACINONYX - Mechanical Linkage Simulation"
if [ "$SKIP_INSTALL" = true ]; then
    echo -e "    ${YELLOW}(quick mode - skipping install checks)${NC}"
fi
echo -e "${NC}"

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
port_available() {
    ! lsof -i:$1 >/dev/null 2>&1
}

# Stream a log file with prefix, filtering empty lines
stream_log() {
    local file="$1"
    local prefix="$2"
    local color="$3"
    if [ -f "$file" ]; then
        tail -f -n 0 "$file" | awk 'NF' | while IFS= read -r line; do
            echo -e "${color}[${prefix}]${NC} $line"
        done
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1

    print_status "Waiting for $name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_status "$name is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    print_error "$name failed to start within 30 seconds"
    return 1
}

# Function to cleanup background processes
cleanup() {
    print_status "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        print_status "Backend server stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_status "Frontend server stopped"
    fi
    if [ ! -z "$LOG_BACKEND_STREAM_PID" ]; then
        kill $LOG_BACKEND_STREAM_PID 2>/dev/null || true
    fi
    if [ ! -z "$LOG_FRONTEND_STREAM_PID" ]; then
        kill $LOG_FRONTEND_STREAM_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_status "Starting Automata project..."

# Step 1: Check Configuration and Environment
print_status "Step 1: Checking configuration and environment..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    print_error "Please run this script from the automata project root directory"
    exit 1
fi

# Check Python
if ! command_exists python; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Check Node.js
if ! command_exists node; then
    print_error "Node.js is not installed or not in PATH"
    exit 1
fi

NODE_VERSION=$(node --version)
print_status "Node.js version: $NODE_VERSION"

# Check npm
if ! command_exists npm; then
    print_error "npm is not installed or not in PATH"
    exit 1
fi

NPM_VERSION=$(npm --version)
print_status "npm version: $NPM_VERSION"

# Check if ports are available
if ! port_available $BACKEND_PORT; then
    print_warning "Port $BACKEND_PORT is already in use. Attempting to kill existing process..."
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
    if ! port_available $BACKEND_PORT; then
        print_error "Unable to free port $BACKEND_PORT"
        exit 1
    fi
fi

if ! port_available $FRONTEND_PORT; then
    print_warning "Port $FRONTEND_PORT is already in use. Attempting to kill existing process..."
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
    if ! port_available $FRONTEND_PORT; then
        print_error "Unable to free port $FRONTEND_PORT"
        exit 1
    fi
fi

# Check if automata package is installed (skip with -q/--quick)
if [ "$SKIP_INSTALL" = false ]; then
    if ! python -c "import automata" 2>/dev/null; then
        print_status "Installing automata package in development mode..."
        pip install -e . || {
            print_error "Failed to install automata package"
            exit 1
        }
    fi

    # Check frontend dependencies
    if [ ! -d "frontend/node_modules" ]; then
        print_status "Installing frontend dependencies..."
        cd frontend
        npm install || {
            print_error "Failed to install frontend dependencies"
            exit 1
        }
        cd ..
    fi
else
    print_status "Skipping dependency checks (quick mode)"
fi

print_status "Environment check completed successfully!"
echo

# Step 2: Start Backend Server
print_status "Step 2: Starting backend server..."
print_status "Backend will run on: http://localhost:$BACKEND_PORT"

# Start backend in background with logging
python -m uvicorn backend.acinonyx_api:app --host 0.0.0.0 --port $BACKEND_PORT --reload > backend.log 2>&1 &
BACKEND_PID=$!

print_status "Backend server started with PID: $BACKEND_PID"
print_status "Backend logs are being written to: backend.log"

# Show backend logs in real-time (first few lines)
print_status "Backend startup logs:"
echo -e "${BLUE}----------------------------------------${NC}"
sleep 2
tail -10 backend.log | awk 'NF' | while read line; do
    echo -e "${BLUE}[BACKEND]${NC} $line"
done
echo -e "${BLUE}----------------------------------------${NC}"

# Wait for backend to be ready
if ! wait_for_service "http://localhost:$BACKEND_PORT/docs" "Backend API"; then
    print_error "Backend failed to start"
    cleanup
    exit 1
fi

# Step 3: Start Frontend
print_status "Step 3: Starting frontend development server..."
print_status "Frontend will run on: $FRONTEND_URL"

cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

print_status "Frontend server started with PID: $FRONTEND_PID"
print_status "Frontend logs are being written to: frontend.log"

# Show frontend logs
print_status "Frontend startup logs:"
echo -e "${GREEN}----------------------------------------${NC}"
sleep 3
tail -10 frontend.log | awk 'NF' | while read line; do
    echo -e "${GREEN}[FRONTEND]${NC} $line"
done
echo -e "${GREEN}----------------------------------------${NC}"

# Wait for frontend to be ready
if ! wait_for_service "$FRONTEND_URL" "Frontend"; then
    print_error "Frontend failed to start"
    cleanup
    exit 1
fi

# Step 4: Open Browser
print_status "Step 4: Opening web browser..."

# Detect OS and open browser
if command_exists open; then
    # macOS
    open "$FRONTEND_URL"
    print_status "Browser opened on macOS"
elif command_exists xdg-open; then
    # Linux
    xdg-open "$FRONTEND_URL"
    print_status "Browser opened on Linux"
elif command_exists start; then
    # Windows
    start "$FRONTEND_URL"
    print_status "Browser opened on Windows"
else
    print_warning "Could not detect how to open browser on this system"
    print_status "Please manually open: $FRONTEND_URL"
fi

echo
print_status "🎉 Automata is now running!"
echo -e "${GREEN}Frontend: ${FRONTEND_URL}${NC}"
echo -e "${GREEN}Backend API: http://localhost:${BACKEND_PORT}${NC}"
echo -e "${GREEN}API Docs: http://localhost:${BACKEND_PORT}/docs${NC}"
if [ "$SKIP_INSTALL" = false ]; then
    echo -e "${YELLOW}Tip: Use './run.sh -q' or './run.sh --quick' to skip install checks${NC}"
fi
echo
print_status "Monitoring logs... Press Ctrl+C to stop all services"
echo

# Stream logs once (filtered), keep script alive to allow Ctrl+C
stream_log backend.log BACKEND "$BLUE" &
LOG_BACKEND_STREAM_PID=$!
stream_log frontend.log FRONTEND "$GREEN" &
LOG_FRONTEND_STREAM_PID=$!

# Wait for background streams
wait
