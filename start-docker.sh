#!/bin/bash

# BioGraphX Docker Startup Script
# Quick start script for running BioGraphX with Docker

set -e

echo "========================================="
echo "   BioGraphX Docker Setup"
echo "========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] Docker Compose is not installed"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "[ERROR] Docker daemon is not running"
    echo "Please start Docker Desktop or Docker daemon"
    exit 1
fi

echo "[OK] Docker is installed and running"
echo ""

# Create necessary directories
echo "[INFO] Creating necessary directories..."
mkdir -p data/chroma data/processed models app/static/graph
echo "[OK] Directories created"
echo ""

# Build and start containers
echo "[INFO] Building Docker images..."
echo "This may take several minutes on first run..."
echo ""

docker-compose build

echo ""
echo "[OK] Build complete!"
echo ""
echo "[INFO] Starting BioGraphX application..."
echo ""

docker-compose up -d

echo ""
echo "[INFO] Waiting for application to start..."
sleep 10

# Check if container is running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "========================================="
    echo "   SUCCESS: BioGraphX is running!"
    echo "========================================="
    echo ""
    echo "Web Interface: http://localhost:1919"
    echo "Health Check:  http://localhost:1919/health"
    echo ""
    echo "Useful commands:"
    echo "  - View logs:        docker-compose logs -f"
    echo "  - Stop application: docker-compose down"
    echo "  - Restart:          docker-compose restart"
    echo ""
    echo "Full documentation: See DOCKER.md"
    echo ""
else
    echo ""
    echo "[ERROR] Container failed to start"
    echo "Check logs with: docker-compose logs"
    echo ""
    exit 1
fi
