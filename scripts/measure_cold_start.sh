#!/bin/bash
IMAGE_NAME="classifier-serving:multi-stage"

docker rm -f classifier-serving 2>/dev/null

echo "Starting cold start measurement..."
START_TIME=$(date +%s%3N)

docker run -d \
  --name classifier-serving \
  -e DOCKER_START_TIME=$START_TIME \
  -p 8000:8000 \
  $IMAGE_NAME

echo "Waiting for application to become healthy..."

while true; do
    if docker logs classifier-serving 2>&1 | grep -q "Application startup complete."; then
        END_TIME=$(date +%s%3N)
        break
    fi
    sleep 0.05
done

CONTAINER_START_ISO=$(docker inspect --format='{{.State.StartedAt}}' classifier-serving)
CONTAINER_START_TIME=$(date -d "$CONTAINER_START_ISO" +%s%3N)

# 6. Calculate Metrics
TOTAL_COLD_START=$((END_TIME - START_TIME))
PULL_AND_PROVISION=$((CONTAINER_START_TIME - START_TIME))
APP_INITIALIZATION=$((END_TIME - CONTAINER_START_TIME))

echo "==========================================="
echo "Cold Start Breakdowns:"
echo "==========================================="
echo "1. Image Pull & Infrastructure Setup: ${PULL_AND_PROVISION} ms"
echo "2. Container Engine to App Ready:    ${APP_INITIALIZATION} ms"
echo "-------------------------------------------"
echo "TOTAL COLD START LATENCY:            ${TOTAL_COLD_START} ms"
echo "==========================================="