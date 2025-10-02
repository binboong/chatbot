#!/bin/bash

# run.sh - Script chạy các lệnh thông dụng

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found!"
        print_info "Creating sample .env file..."
        cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key_here
EOF
        print_info "Please edit .env file and add your Google API key"
        exit 1
    fi
}

# Setup database
setup_db() {
    print_info "Setting up vector database..."
    python db_setup.py
    print_success "Database setup completed!"
}

# Ingest documents
ingest_docs() {
    if [ "$1" = "--force" ]; then
        print_info "Force ingesting documents..."
        python ingest.py --force
    else
        print_info "Ingesting new documents..."
        python ingest.py
    fi
    print_success "Document ingestion completed!"
}

# Run chatbot
run_chatbot() {
    print_info "Starting chatbot..."
    python chatbot.py
}

# Run API server
run_api() {
    print_info "Starting API server..."
    python api.py
}

# Build Docker image
build_docker() {
    print_info "Building Docker image..."
    docker-compose build
    print_success "Docker image built successfully!"
}

# Run with Docker
run_docker() {
    check_env_file
    print_info "Starting services with Docker..."
    docker-compose up -d
    print_success "Services started! API available at http://localhost:8000"
    
    # Show logs
    if [ "$1" = "--logs" ]; then
        docker-compose logs -f
    fi
}

# Stop Docker services
stop_docker() {
    print_info "Stopping Docker services..."
    docker-compose down
    print_success "Services stopped!"
}

# Install dependencies
install_deps() {
    print_info "Installing Python dependencies..."
    pip install -r requirements.txt
    print_success "Dependencies installed!"
}

# Show help
show_help() {
    echo "🤖 Chatbot Management Script"
    echo ""
    echo "Usage: ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup          - Setup vector database"
    echo "  ingest         - Ingest documents from data/ folder"
    echo "  ingest --force - Force re-ingest all documents"
    echo "  chatbot        - Run interactive chatbot"
    echo "  api            - Run FastAPI server"
    echo "  docker:build   - Build Docker image"
    echo "  docker:run     - Run with Docker Compose"
    echo "  docker:logs    - Run with Docker and show logs"
    echo "  docker:stop    - Stop Docker services"
    echo "  install        - Install Python dependencies"
    echo "  help           - Show this help"
    echo ""
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh ingest"
    echo "  ./run.sh chatbot"
    echo "  ./run.sh docker:run"
}

# Main logic
case "$1" in
    setup)
        check_env_file
        setup_db
        ;;
    ingest)
        check_env_file
        ingest_docs $2
        ;;
    chatbot)
        check_env_file
        run_chatbot
        ;;
    api)
        check_env_file
        run_api
        ;;
    docker:build)
        check_env_file
        build_docker
        ;;
    docker:run)
        run_docker
        ;;
    docker:logs)
        run_docker --logs
        ;;
    docker:stop)
        stop_docker
        ;;
    install)
        install_deps
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac