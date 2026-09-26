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

## Load test
Run load tests via Locust from the classifier package root
```
locust -f load-tests/locustfile.py -H http://localhost:8000 --headless -u <number of users> -r <number of users> -t 1m --csv load-tests/<number of users>_users/<number of users>_users
```
For example to run a load test with 8 concurrent users, use the following command
```
locust -f load-tests/locustfile.py -H http://localhost:8000 --headless -u 8 -r 8 -t 1m --csv load-tests/8_users/8_users
```