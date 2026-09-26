# Notes

## Different ways of calling top-level torch functions in Pytorch
Pytorch provides different ways of calling top-level torch functions like `softmax` and `argmax`.
It can be called as a standard PyTorch function via `torch.softmax(logits)` by passing the tensor as the first argument
or called directly on a tensor object as `logits.softmax()` for e.g.

Both perform the exact same mathematical operation and return identical results.

Pytorch maps many of its top-level torch functions like `.softmax()` and `argmax()` directly onto the 
`torch.Tensor` class as methods. This allows easy chaining of operations together.

## `pip` vs `uv` for Docker image building
`uv` Python Dockerfiles are significantly faster than traditional `pip` Dockerfiles due at resolving and downloading dependencies due to being written in Rust, utilizing aggressive parallel execution and uses an advanced dependency solver.

## Docker image optimization
Before Optimization:
- Content Size: `3.14GB`
After Optimization (multistage build):
- Content Size: `3.12GB`

Expected given that CUDA-enabled torch packages (which are a few GBs in size) are installed.

## Image cold start time
Cold start time recorded only for multistage image, I do not expect much difference in cold start time between the non multistage image and the multistage image given that the multistage image is only 0.02 GB smaller than the non multistage image.

| Cold Start Breakdowns: |
| :----------------------|
| Image Pull & Infrastructure Setup: 59 ms |
| Container Engine to App Ready: 8522 ms |
| TOTAL COLD START LATENCY: 8581 ms |

## Load test

### CPU

Load tested the predict endpoint via `locust` at concurrency 1, 8 and 32 for 1 minute each. Each load test spawns the exact number of workers so there is no ramp up.

| | 1 worker| 8 workers | 32 workers |
| :--- | :---: | :---: | :---: |
| **Request Count** | 6152 | 8373 | 7823 |
| **Requests/s** | 104~ | 141~ | 132~ |
| **p50 latency** | 9 | 56 | 240 |
| **p95 latency** | 10 | 64 | 270 |
| **p99 latency** | 12 | 67 |  290 |

Since the load tests take place in a closed-loop where each user waits for a response before sending the next request, Little's law holds approximately:
```
concurrency ≈ throughput x latency
```

| Users | Requests/s | Predicted mean latency (users ÷ req/s) | Actual p99 |
| :---: | :---: | :---: | :---: |
| 1 | ~103 | ~10 ms | 12 ms |
| 8 | ~140 | ~57 ms | 67 ms |
| 32 | ~131 | ~245 ms | 290 ms |

The predicted mean latency comes out to be just below the actual p99 at every level so we can conclude that the server tops out at about **140 requests per second**. Beyond that having more users just result in longer waiting time for access to resources.