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


### GPU

Same set of load tests ran except GPU is enabled

#### Setup
- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB, driver 580, compute capability 12.0 (Blackwell).
- The PyPI `torch` wheels bundle their own CUDA runtime (the `nvidia-*` packages in `uv.lock`, CUDA 13), so the image does not need an `nvidia/cuda` base image. The host only provides the driver.
- The host needs the NVIDIA Container Toolkit installed and Docker configured to use it, then the container is run with `--gpus all`.
- The model is moved to the device once at startup in `lifespan`, and only the input tensors are moved per request. Choosing the device per request is unnecessary since the answer cannot change while the process runs.
- `torch.cuda.is_available()` falls back to CPU silently. If the container is started without `--gpus all`, everything still works but on CPU, so a "GPU" load test would silently produce CPU numbers.

| | 1 user| 8 users | 32 users |
| :--- | :---: | :---: | :---: |
| **Request Count** | 27153 | 27025 | 25380 |
| **Requests/s** | 460~ | 458~ | 430~ |
| **p50 latency** | 2 | 17 | 74 |
| **p95 latency** | 2 | 19 | 82 |
| **p99 latency** | 3 | 20 |  90 |

We can clearly see that the usage of GPU has greatly improved the throughput and latency.
However, based on Little's law, we can see that increasing the number of users (concurrency) does not do anything to improve the performance.
```
concurrency ≈ throughput x latency

1 user:
1 / 460 = ~2.2 ms
8 users:
8 / 458 = ~17.5 ms
32 users:
32 / 430 = ~74.4 ms
```
The predicted mean latencies are very close to the actual p50 latencies recorded. We can conclude that the GPU is not the bottleneck. The bottleneck is probably the Python process that is called everytime the predict endpoint is called.

Unlike the CPU run, where going from 1 to 8 users still gave ~1.4× more throughput, on GPU going from 1 to 8 users gives **no** extra throughput. The server is already saturated at 1 user, so from 8 users onwards the latency is almost entirely queue time. The small drop from 8 to 32 users is consistent with more threads competing for the GIL.

#### What the hardware was doing (1 worker, 32 users)

| | Observed | Meaning |
| :--- | :--- | :--- |
| `nvidia-smi` GPU-Util | 29% | GPU mostly idle |
| `nvidia-smi` power | 76 W / 300 W | GPU mostly idle |
| `nvidia-smi` memory | 1840 MiB | Model weights (~260 MB fp32) + CUDA context + PyTorch's memory cache |
| `docker stats` CPU | ~219% (~2.2 of 16 cores) | Server far from CPU-bound |
| `docker stats` PIDs | 91 | Mostly threads: thread pool, PyTorch threads, uvicorn |

Neither the GPU nor the CPU is saturated, yet throughput is flat from 1 to 32 users, so the limit is **serialization within one process**:
- The GIL: HTTP handling, pydantic, the handler and calls into PyTorch run one at a time. CPU goes above 100% only because some work releases the GIL (Rust tokenizer, PyTorch ops, CUDA sync).
- All threads send GPU work to the same default CUDA stream, so their kernels also run one after another.

GPU-Util measures the share of time *any* kernel was running, not how much of the GPU each kernel uses. A tiny DistilBERT forward pass uses a small part of the chip, so real compute use is far below the number shown.

Locust ruled out as the bottleneck: it ran as a single process (no `--processes`) and the "CPU usage above 90%" warning never appeared. Caveat: the warning is a sampled heuristic, and Locust shares the same 16 cores as the server.

### GPU with multiple FastAPI workers

32 users and 2 FastAPI workers
```
Requests Count: 46837
Requests/s: 793~
p50: 39 ms
p90: 51 ms
p99: 55 ms
GPU utlization: 93%
server CPU utilization: 503%
```

32 users and 4 FastAPI workers
```
Requests Count: 65991
Requests/s: 1118~
p50: 27 ms
p90: 38 ms
p99: 46 ms
GPU utlization: 93%
server CPU utilization: 1222%
```

| Workers (32 users) | Requests/s | vs. 1 worker | Predicted mean (32 ÷ req/s) | Actual p50 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | ~430 | 1.0× | ~74 ms | 74 ms |
| 2 | ~793 | 1.84× | ~40 ms | 39 ms |
| 4 | ~1118 | 2.6× | ~29 ms | 27 ms |

- 1 → 2 workers is almost linear, which confirms that one process was the ceiling. Each worker is a separate process with its own GIL, Python interpreter and CUDA context.
- 2 → 4 workers gives only 1.41×, so a new limit starts to apply.
- Little's law still matches p50, so at 32 users latency is still mostly queue time. The model itself takes ~2 ms (see the 1 user result).

CPU time per request (server CPU ÷ requests/s):

| Workers | Server CPU | CPU time per request |
| :---: | :---: | :---: |
| 1 | ~2.2 cores | ~5.1 ms |
| 2 | ~5.0 cores | ~6.3 ms |
| 4 | ~12.2 cores | ~10.9 ms |

Each request costs about twice as much CPU at 4 workers as at 1 worker even though the work did not change, so the extra is contention overhead. At ~12 of 16 cores (with Locust sharing the same cores), 8 workers would run out of CPU.

GPU-Util shows 93% at both 2 and 4 workers, yet throughput still rose 40%. This confirms GPU-Util is not a capacity measure. Multiple processes do not run kernels on the GPU at the same time, the GPU switches between them, so it is "busy" nearly all the time with many tiny kernels while its real compute is still mostly unused.

GPU memory per worker (4 workers, snapshot while idle): ~872 MiB each, found via `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`. Each worker loads its own copy of the model, so 16 GB fits several workers.

**One worker vs. many (Break and fix item):** with one worker, the ceiling is the single Python process (GIL + serialized launches), so both GPU and CPU are underused. More workers parallelize the per-request CPU work across processes, which raises throughput until CPU contention and GPU switching between processes cause diminishing returns.

### GPU with multiple FastAPI Workers and limit PyTorch to two threads per worker

32 users, 4 FastAPI workers and limit pytorch to 2 threads
```
Requests Count: 65820
Requests/s: 1114~
p50: 28 ms
p90: 34 ms
p99: 40 ms
GPU utlization: 93%
server CPU utilization: 1175%
```

| 4 workers, 32 users | Requests/s | p50 | p99 | Server CPU | CPU time per request |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Default threads | ~1118 | 27 ms | 46 ms | ~12.2 cores | ~10.9 ms |
| 2 PyTorch threads | ~1114 | 28 ms | 40 ms | ~11.8 cores | ~10.5 ms |

Null result: limiting PyTorch threads changed almost nothing. PyTorch's intra-op threads parallelize work **on the CPU**, but the forward pass runs on the GPU, so those threads have little to do. Thread oversubscription mattered for the CPU-only setup, but it is not where the CPU time goes now.

Caveats:
- One run per configuration, so differences this small (e.g. the 6 ms p99 change) may be noise. Run each configuration 2–3 times to know the noise band.
- Not yet verified that the setting applied. `OMP_NUM_THREADS` only takes effect if set before `torch` is imported, and each worker is its own process. Check `torch.get_num_threads()` inside a worker.

### Profiling a worker with py-spy

Finding the PIDs to profile (from the host):
- `docker top classifier-serving` lists the container's processes with **host** PIDs. The container has its own PID namespace, so `ps` inside the container shows different numbers.
- With `--workers 4` there is a supervisor (`fastapi run ... --workers 4`, serves no requests), a multiprocessing resource tracker, and 4 workers (`spawn_main ... --multiprocessing-fork`).
- `nvidia-smi --query-compute-apps=...` lists only the workers, since only they load the model onto the GPU.
- PIDs change every time the container restarts.

Running py-spy:
- Installed on the host, not in the image.
- Needs `sudo env "PATH=$PATH" py-spy ...` because the container processes run as root, and `sudo` resets `PATH` so a user-installed py-spy is not found otherwise.
- `py-spy record --native` failed with `UNW_EBADREG: bad register number`. This comes from libunwind failing to walk a native frame (PyTorch, CUDA, cuDNN, Rust tokenizer libraries are all loaded). Without `--native` it works, with time inside native code attributed to the Python line that called it.

Flame graph of one worker under 4 workers / 32 users (`py-spy record --pid <worker pid>`):

| Block | Approx. width | What it is |
| :--- | :---: | :--- |
| `spawn_main` → uvicorn → asyncio run loop | ~9% | Event loop: connections, HTTP parsing, FastAPI request handling |
| anyio worker thread → `predict (main.py:57)` → model `forward` | ~75% | Sync handler in the thread pool, almost all inside the DistilBERT forward pass |
| Inside `predict`, outside `forward` | a few % | Tokenizing, `.to(device)`, softmax, `.item()` |
| `always_pong` / `recv` | ~9% | uvicorn heartbeat between supervisor and worker, not request work |

- The model forward call dominates the CPU, not HTTP, pydantic or the tokenizer. Cutting per-request web overhead would not achieve much.
- That CPU time is **dispatch, not computation**: walking nested Python module calls (`_wrapped_call_impl` → `_call_impl` → `forward` at every layer, plus `transformers` wrapper frames) and launching hundreds of small kernels. CUDA kernel launches return immediately and the GPU finishes them quickly, so the CPU is struggling to send kernels fast enough.
- `.item()` (where the CPU waits for the GPU) is narrow, so the earlier hypothesis that CPU time was spent spinning while waiting on the GPU was mostly wrong.
- This explains the earlier results: more workers helped because launch overhead is CPU work that can run in parallel across processes, and GPU-Util was high because of a constant stream of tiny kernels.
- Caveats: one worker profiled under 4-worker load, so it does not explain why CPU per request doubled from 1 to 4 workers (comparing with a 1-worker profile would). "Launch overhead" is inferred, not proven, since `--native` failed.

### Next: request batching

Model the time for one forward pass on a batch of B requests as:
```
batch time ≈ F + B × m
throughput ≈ B / (F + B × m)    → approaches 1 / m as B grows
```
- **F** is the fixed cost per call: Python module dispatch, kernel launches, synchronization, reading weights from GPU memory.
- **m** is the marginal cost per extra item: the actual computation for that input.

The profile shows the server is far into the "F dominates" regime, so batching should raise throughput a lot by paying F once per batch instead of once per request.

Costs of batching: waiting for a batch to fill (pure added latency at low load), waiting for the whole batch to finish, padding to the longest input in the batch (hidden by the fixed test prompt), and head-of-line blocking. At high load batching is almost free because requests are queueing anyway.

Settings to tune: maximum batch size, maximum wait (timeout), and worker count. With per-worker batching each worker sees a fraction of the traffic, so once batching removes most of the launch overhead, fewer workers with fuller batches may beat more workers with small ones.

Plan: measure the offline batch-cost curve (forward pass time for B = 1, 2, 4, …, 128, no HTTP) first, then adopt LitServe (check Python 3.14 support), and rerun the 1/8/32 user tests with and without batching, including the 1 user p50 to show batching's cost.