#!/bin/bash
set -euo pipefail

# PINN Fixes Deployment Script
# Supports both canary and blue-green deployment strategies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOYMENT_DIR")"

# Configuration
NAMESPACE="options-production"
IMAGE_TAG="${IMAGE_TAG:-pinn-fixes-v1.0.0}"
STRATEGY="${STRATEGY:-canary}"  # canary or blue-green
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        error "Namespace $NAMESPACE does not exist"
    fi
    
    # Check if image exists
    log "Checking if image exists: options-optimizer:$IMAGE_TAG"
    if ! docker image inspect "options-optimizer:$IMAGE_TAG" &>/dev/null; then
        warn "Image options-optimizer:$IMAGE_TAG not found locally"
        log "Building image..."
        docker build -t "options-optimizer:$IMAGE_TAG" -f "$DEPLOYMENT_DIR/staging/Dockerfile.staging" "$PROJECT_ROOT"
    fi
    
    # Check Argo Rollouts is installed
    if ! kubectl get crd rollouts.argoproj.io &>/dev/null; then
        error "Argo Rollouts CRD not found. Please install Argo Rollouts first."
    fi
    
    success "Pre-deployment checks passed"
}

# Run staging tests
run_staging_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping staging tests (SKIP_TESTS=true)"
        return 0
    fi
    
    log "Running staging tests..."
    
    # Start staging environment
    log "Starting staging environment..."
    cd "$DEPLOYMENT_DIR/staging"
    docker-compose -f docker-compose.staging.yml up -d
    
    # Wait for staging to be ready
    log "Waiting for staging environment to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:8001/health/deployment/readiness | grep -q '"deployment_ready":true'; then
            success "Staging environment is ready"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "Staging environment failed to become ready after $max_attempts attempts"
        fi
        
        log "Attempt $attempt/$max_attempts - waiting for staging readiness..."
        sleep 10
        ((attempt++))
    done
    
    # Run PINN performance tests
    log "Running PINN performance tests..."
    python3 "$PROJECT_ROOT/scripts/performance_benchmark.py" --target-url http://localhost:8001 --target-latency 260
    
    # Run load tests
    log "Running load tests..."
    "$SCRIPT_DIR/load-test-pinn.sh" http://localhost:8001
    
    # Cleanup staging
    log "Cleaning up staging environment..."
    docker-compose -f docker-compose.staging.yml down
    
    success "Staging tests completed successfully"
}

# Deploy using canary strategy
deploy_canary() {
    log "Deploying PINN fixes using canary strategy..."
    
    # Apply analysis templates
    kubectl apply -f "$DEPLOYMENT_DIR/rollout/analysis-templates.yaml"
    
    # Apply canary deployment
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f "$DEPLOYMENT_DIR/rollout/canary-deployment.yaml"
        log "Dry run completed - no changes applied"
        return 0
    fi
    
    kubectl apply -f "$DEPLOYMENT_DIR/rollout/canary-deployment.yaml"
    
    # Monitor rollout
    log "Monitoring canary rollout..."
    kubectl argo rollouts get rollout options-api-pinn-fixes -n "$NAMESPACE" --watch
}

# Deploy using blue-green strategy
deploy_blue_green() {
    log "Deploying PINN fixes using blue-green strategy..."
    
    # Apply analysis templates
    kubectl apply -f "$DEPLOYMENT_DIR/rollout/analysis-templates.yaml"
    
    # Apply blue-green deployment
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f "$DEPLOYMENT_DIR/rollout/blue-green-deployment.yaml"
        log "Dry run completed - no changes applied"
        return 0
    fi
    
    kubectl apply -f "$DEPLOYMENT_DIR/rollout/blue-green-deployment.yaml"
    
    # Monitor rollout
    log "Monitoring blue-green rollout..."
    kubectl argo rollouts get rollout options-api-blue-green -n "$NAMESPACE" --watch
}

# Main deployment function
main() {
    log "Starting PINN fixes deployment"
    log "Strategy: $STRATEGY"
    log "Image Tag: $IMAGE_TAG"
    log "Namespace: $NAMESPACE"
    log "Dry Run: $DRY_RUN"

    # Run pre-deployment checks
    pre_deployment_checks

    # Run staging tests
    run_staging_tests

    # Deploy based on strategy
    case "$STRATEGY" in
        "canary")
            deploy_canary
            ;;
        "blue-green")
            deploy_blue_green
            ;;
        *)
            error "Unknown deployment strategy: $STRATEGY. Use 'canary' or 'blue-green'"
            ;;
    esac

    success "PINN fixes deployment initiated successfully"
    log "Monitor deployment progress with:"
    log "  kubectl argo rollouts get rollout options-api-$STRATEGY -n $NAMESPACE --watch"
    log ""
    log "To promote (blue-green only):"
    log "  kubectl argo rollouts promote options-api-blue-green -n $NAMESPACE"
    log ""
    log "To abort deployment:"
    log "  kubectl argo rollouts abort options-api-$STRATEGY -n $NAMESPACE"
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --strategy STRATEGY     Deployment strategy (canary|blue-green) [default: canary]"
            echo "  --image-tag TAG         Docker image tag [default: pinn-fixes-v1.0.0]"
            echo "  --namespace NAMESPACE   Kubernetes namespace [default: options-production]"
            echo "  --dry-run              Run in dry-run mode (no changes applied)"
            echo "  --skip-tests           Skip staging tests"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --strategy canary --image-tag v1.0.1"
            echo "  $0 --strategy blue-green --dry-run"
            echo "  $0 --skip-tests --namespace options-staging"
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Run main function
main
}
