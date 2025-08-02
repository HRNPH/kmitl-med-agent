#!/bin/bash

# Docker management scripts for the FastAPI application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build and start the application
start() {
    print_status "Starting the application..."
    check_docker
    
    # Create data directory if it doesn't exist
    mkdir -p ./data
    
    # Build and start services
    docker-compose up --build -d
    
    print_status "Application is starting up..."
    print_status "FastAPI docs will be available at: http://localhost:8000/docs"
    print_status "Ollama will be available at: http://localhost:11434"
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are healthy
    if docker-compose ps | grep -q "healthy"; then
        print_status "All services are healthy!"
    else
        print_warning "Some services may still be starting up. Check with: docker-compose ps"
    fi
}

# Function to stop the application
stop() {
    print_status "Stopping the application..."
    docker-compose down
    print_status "Application stopped."
}

# Function to restart the application
restart() {
    print_status "Restarting the application..."
    stop
    start
}

# Function to view logs
logs() {
    if [ -z "$1" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$1"
    fi
}

# Function to pull the latest Ollama model
pull_model() {
    if [ -z "$1" ]; then
        print_error "Please specify a model name. Usage: $0 pull-model <model_name>"
        print_status "Example: $0 pull-model qwen2:7b"
        exit 1
    fi
    
    print_status "Pulling Ollama model: $1"
    docker-compose exec ollama ollama pull "$1"
}

# Function to list available models
list_models() {
    print_status "Available Ollama models:"
    docker-compose exec ollama ollama list
}

# Function to clean up everything
cleanup() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_status "Cleanup completed."
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to show status
status() {
    print_status "Service status:"
    docker-compose ps
    
    echo ""
    print_status "Container logs (last 10 lines):"
    docker-compose logs --tail=10
}

# Function to show help
help() {
    echo "Usage: $0 {start|stop|restart|logs|status|pull-model|list-models|cleanup|help}"
    echo ""
    echo "Commands:"
    echo "  start         - Build and start the application"
    echo "  stop          - Stop the application"
    echo "  restart       - Restart the application"
    echo "  logs [service]- View logs (optional service name)"
    echo "  status        - Show service status and recent logs"
    echo "  pull-model    - Pull a specific Ollama model"
    echo "  list-models   - List available Ollama models"
    echo "  cleanup       - Remove all containers, images, and volumes"
    echo "  help          - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs app"
    echo "  $0 pull-model qwen2:7b"
}

# Main script logic
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs "$2"
        ;;
    status)
        status
        ;;
    pull-model)
        pull_model "$2"
        ;;
    list-models)
        list_models
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        help
        exit 1
        ;;
esac 