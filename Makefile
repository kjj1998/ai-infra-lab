# Variables
IMAGE_NAME := classifier-serving
DOCKERFILE_PATH := packages/classifier/Dockerfile
CONTEXT_DIR := .

.PHONY: all build run run-gpu dev

# Default target
all: build

# Dev
dev: 
	uv run --env-file .env python -m classifier.main 

# Run classifier tests
test-classifier:
	uv run pytest packages/classifier/tests/unit/* -v

## build: Build the Docker image for the classifier serving layer
build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE_PATH) $(CONTEXT_DIR)

## run: Run the Docker image for the classifier serving layer using CPU
run:
	docker run --env-file .env -d --name $(IMAGE_NAME) -p 8000:8000 $(IMAGE_NAME):latest

## run-gpu: Run the Docker image for the classifier serving layer using GPU
run-gpu:
	docker run --env-file .env -d --gpus 'all' --name $(IMAGE_NAME) -p 8000:8000 $(IMAGE_NAME):latest

