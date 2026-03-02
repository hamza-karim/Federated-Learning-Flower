#!/bin/bash
# build_docker_images.sh - Build images for both platforms

set -e  # Exit on error

# ============================================
# Configuration
# ============================================
DOCKER_REGISTRY="hamzakarim07"
PROJECT_NAME="flwr_client"

VERSION="${1:-latest}"  # Default to 'latest' if no version specified

# ============================================
# Build ARM64 Image (for Jetson AGX)
# ============================================
echo "=========================================="
echo "Building ARM64 Image for Jetson AGX"
echo "=========================================="

docker build \
    --platform linux/arm64 \
    -f FL_client/docker/Dockerfile.arm64 \
    -t ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:${VERSION} \
    -t ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:arm64-${VERSION} \
    .

echo "✅ ARM64 image built: ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:${VERSION}"

# ============================================
# Build x86_64 Image (for Lambda Server)
# ============================================
echo ""
echo "=========================================="
echo "Building x86_64 Image for Lambda Server"
echo "=========================================="

docker build \
    --platform linux/amd64 \
    -f FL_client/docker/Dockerfile.x86_64 \
    -t ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:${VERSION} \
    -t ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:x86_64-${VERSION} \
    .

echo "✅ x86_64 image built: ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:${VERSION}"

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "BUILD COMPLETE"
echo "=========================================="
echo "Images created:"
echo "  ARM64 (Jetson AGX):"
echo "    - ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:${VERSION}"
echo "    - ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:arm64-${VERSION}"
echo ""
echo "  x86_64 (Lambda Server):"
echo "    - ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:${VERSION}"
echo "    - ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:x86_64-${VERSION}"
echo ""
echo "To push to Docker Hub:"
echo "  docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:${VERSION}"
echo "  docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:${VERSION}"
echo "=========================================="

# ============================================
# Optional: Push to Registry
# ============================================
read -p "Push images to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing ARM64 image..."
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:${VERSION}
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_hfl:arm64-${VERSION}
    
    echo "Pushing x86_64 image..."
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:${VERSION}
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}_lambda:x86_64-${VERSION}
    
    echo "✅ All images pushed successfully!"
fi