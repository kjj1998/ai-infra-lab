# Classifier

# Intro
This section contains information about the text classifier part of the repository which I am using to learn AI Infrastructure.

# Start Up
Run `uv run python -m classifier.main`

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

Run the image with GPUs enabled (must install Nvidia Container Runtime Kit, see section below)
```
docker run -d --gpus 'all' --name classifier-serving -p 8000:8000 classifier-serving:latest
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

## Setting up Nvidia Container Runtime Kit

To access GPUs from within Docker containers, the Nvidia Container Runtime Kit must be installed.

To install the Nvidia Container Runtime Kit, follow the steps at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Configure the installed Nvidia Container Runtime Kit
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test that `nvidia-smi` can be accessed from within a container
```
docker run -it --rm --gpus 'all,driver=nvidia' ubuntu nvidia-smi
```