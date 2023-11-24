#!/bin/bash

# Define the Docker image name and tag
IMAGE_NAME="pixelda"
TAG="latest"
DOCKER_IMAGE="${IMAGE_NAME}:${TAG}"

# Check if the Docker image exists
if ! docker inspect "${DOCKER_IMAGE}" &> /dev/null; then
    echo "Docker image '${DOCKER_IMAGE}' not found. Creating the image..."
    docker build -t "${DOCKER_IMAGE}" docker/
fi

# Check if the path parameter is provided
if [ "$#" -eq 0 ]; then
    echo "Error: Path parameter not provided. Usage: $0 --config-file /path/to/their/test_config_file_name"
    exit 1
fi

PIXELDA_DIR="$(pwd)"
echo "$(pwd)"
docker run -it --gpus all -v "${PIXELDA_DIR}:/pixelda" "${DOCKER_IMAGE}" python3 test.py "$@" 