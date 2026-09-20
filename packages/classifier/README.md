# Classifier

# Intro
This section contains information about the text classifier part of the repository which I am using to learn AI Infrastructure.

# Start Up
Run `uv run fastapi dev`

## Docker
Build the Docker image, from project root run
```
docker build -t classifier-serving -f packages/classifier/Dockerfile .
```

Run the image in detached mode
```
docker run -d --name classifier-serving -p 8000:8000 classifier-serving:latest
```

View the logs from the running container
```
docker logs -f classifier-serving
```