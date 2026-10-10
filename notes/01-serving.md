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

## Load tests

### CPU (ACCELERATOR=cpu)

Every run is 1 minute with the exact number of users spawned with no ramp up

| ID  | Batch        | Timeout | Workers | API | Threads | Users | Purpose                            |
|-----|--------------|---------|---------|-----|---------|-------|------------------------------------|
| C0a | 1            | 0       | 1       | 1   | default | 1     | Baseline vs FastAPI CPU            |
| C0b | 1            | 0       | 1       | 1   | default | 8     |                                    |
| C0c | 1            | 0       | 1       | 1   | default | 32    |                                    |
| C1a | 8            | 0.001   | 1       | 1   | default | 32    | Batching sweep                     |
| C1b | 8            | 0.005   | 1       | 1   | default | 32    |                                    |
| C1c | 8            | 0.01    | 1       | 1   | default | 32    |                                    |
| C1d | 16           | 0.001   | 1       | 1   | default | 32    |                                    |
| C1e | 16           | 0.005   | 1       | 1   | default | 32    |                                    |
| C1f | 16           | 0.01    | 1       | 1   | default | 32    |                                    |
| C1g | 32           | 0.001   | 1       | 1   | default | 32    |                                    |
| C1h | 32           | 0.005   | 1       | 1   | default | 32    |                                    |
| C1i | 32           | 0.01    | 1       | 1   | default | 32    |                                    |
| C2a | best C1      | best C1 | 2       | 1   | default | 32    | Workers + oversubscription         |
| C2b | best C1      | best C1 | 2       | 1   | 6       | 32    |                                    |
| C2c | best C1      | best C1 | 4       | 1   | default | 32    |                                    |
| C2d | best C1      | best C1 | 4       | 1   | 6       | 32    |                                    |
| C3a | best C2      | best C2 | best C2 | 2   | best C2 | 32    | Does HTTP matter on CPU?           |
| C4a | best overall |         |         |     |         | 1     | Final table                        |
| C4b | best overall |         |         |     |         | 8     |                                    |
| C4c | best overall |         |         |     |         | 32    |                                    |

#### Baseline (C0a - C0c)

|                   | 1 worker | 8 workers | 32 workers |
| :---------------- | :------: | :-------: | :--------: |
| **Request Count** |   5609   |   6837    |    6846    |
| **Requests/s**    |   95~    |   116~    |    116~    |
| **p50 latency**   |    10    |    68     |    270     |
| **p95 latency**   |    13    |    73     |    280     |
| **p99 latency**   |    13    |    77     |    300     |


#### With batching (C1a - C1i)

Batch size 8 and 32 users

| Batch Timeout (s) | 0.001 | 0.005 | 0.01  |
| :---------------- | :---: | :---: | :---: |
| **Request Count** | 24295 | 24304 | 24608 |
| **Requests/s**    | 412~  | 412~  | 417~  |
| **p50 latency**   |  77   |  77   |  76   |
| **p95 latency**   |  86   |  86   |  85   |
| **p99 latency**   |  91   |  91   |  90   |

Batch size 16 and 32 users

| Batch Timeout (s) | 0.001 | 0.005 | 0.01  |
| :---------------- | :---: | :---: | :---: |
| **Request Count** | 31255 | 32499 | 32757 |
| **Requests/s**    | 529~  | 548~  | 554~  |
| **p50 latency**   |  60   |  56   |  55   |
| **p95 latency**   |  70   |  68   |  68   |
| **p99 latency**   |  76   |  78   |  73   |

Batch size 32 and 32 users

| Batch Timeout (s) | 0.001 | 0.005 | 0.01  |
| :---------------- | :---: | :---: | :---: |
| **Request Count** | 32193 | 29578 | 31433 |
| **Requests/s**    | 545~  | 500~  | 531~  |
| **p50 latency**   |  57   |  70   |  57   |
| **p95 latency**   |  69   |  86   |  95   |
| **p99 latency**   |  74   |  93   |  110  |


#### With batching, workers and threads (C2a - C2d)

batch size 16, batch timeout 0.01s, 2 workers and 32 users

| Threads           | 0 (default) |   6   |
| :---------------- | :---------: | :---: |
| **Request Count** |    32449    | 34386 |
| **Requests/s**    |    548~     | 582~  |
| **p50 latency**   |     56      |  53   |
| **p95 latency**   |     84      |  62   |
| **p99 latency**   |     95      |  93   |

batch size 16, batch timeout 0.01s, 4 workers and 32 users

| Threads           | 0 (default) |   6   |
| :---------------- | :---------: | :---: |
| **Request Count** |    2201     | 9157  |
| **Requests/s**    |     37~     | 155~  |
| **p50 latency**   |     840     |  190  |
| **p95 latency**   |    1200     |  400  |
| **p99 latency**   |    1400     |  520  |

#### With 2 API servers

|                   | 2 API Servers |
| :---------------- | :-----------: |
| **Request Count** |     33298     |
| **Requests/s**    |     564~      |
| **p50 latency**   |      55       |
| **p95 latency**   |      65       |
| **p99 latency**   |      95       |

#### Best overall settings with different number of users

Batch size 16, batch timeout 0.01s, 2 workers, 1 API server, 6 threads

| Users             |   1   |   8   |  32   |
| :---------------- | :---: | :---: | :---: |
| **Request Count** | 3213  | 15859 | 33160 |
| **Requests/s**    |  54~  | 269~  | 561~  |
| **p50 latency**   |  19   |  30   |  51   |
| **p95 latency**   |  21   |  34   |  71   |
| **p99 latency**   |  22   |  37   |  95   |

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

### LitServe without batching (max batch size 1, 1 inference worker)

Baseline before turning batching on, to measure LitServe's own overhead. Same Locust setup as before (GPU, 1 minute per level, no ramp up), except the endpoint is now `POST /predict` since LitServe only registers POST.

How LitServe is structured (different from `fastapi run`):
- **API server process(es)** run FastAPI/uvicorn, receive HTTP requests and put them on a queue.
- **Inference worker(s)** (`workers_per_device` × devices) take requests off the queue, run `decode_request` → `predict` → `encode_response` one request at a time in a single loop, and send the result back through a response queue.
- With `fastapi run --workers N`, each worker did both jobs in one process. With LitServe they are separate processes, each with its own GIL.

| | 1 user | 8 users | 32 users |
| :--- | :---: | :---: | :---: |
| **Request Count** | 22657 | 35777 | 35916 |
| **Requests/s** | 384~ | 606~ | 608~ |
| **p50 latency** | 3 | 13 | 52 |
| **p95 latency** | 3 | 14 | 55 |
| **p99 latency** | 3 | 16 | 60 |
| **Max latency** | 395 | 28 | 96 |
| **Failures** | 0 | 0 | 0 |

Requests/s is Locust's reported rate, which is slightly higher than request count ÷ 60.

Little's law still holds, so from 8 users onwards latency is mostly queue time:
```
1 user:   1 / 384  = ~2.6 ms   (actual average 2.6 ms)
8 users:  8 / 606  = ~13.2 ms  (actual average 13.2 ms)
32 users: 32 / 608 = ~52.6 ms  (actual average 52.5 ms)
```

Compared with FastAPI on GPU with 1 worker:

| Users | FastAPI req/s | LitServe req/s | FastAPI p50 / p99 | LitServe p50 / p99 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | ~460 | ~384 (−17%) | 2 / 3 ms | 3 / 3 ms |
| 8 | ~458 | ~606 (+32%) | 17 / 20 ms | 13 / 16 ms |
| 32 | ~430 | ~608 (+41%) | 74 / 90 ms | 52 / 60 ms |

**Why LitServe is slower at 1 user:** every request takes an extra hop between processes (API server → request queue → inference worker → response queue → API server). That costs roughly 0.4 ms per request, and at 1 user nothing overlaps with it, so it adds directly to every request.

**Why LitServe is faster at 8 and 32 users, even without batching:**
1. The work runs in parallel across processes. HTTP parsing, pydantic and JSON run in the API server process while the inference worker runs the model, each with its own GIL. With FastAPI they shared one GIL in one process. The py-spy profile put the event loop at ~9% of the time, so this accounts for part of the gain.
2. No thread contention, probably the larger part. With FastAPI, throughput fell from 1 to 32 users (460 → 430) because up to 32 thread pool threads fought over one GIL. LitServe's inference worker processes requests one at a time in a single loop, and throughput is flat from 8 to 32 users (606 → 608).

The new ceiling of ~607 req/s is the inference worker's own time per request (~1.65 ms), which per the py-spy profile is mostly forward pass launch overhead. The inference worker is now the bottleneck, which is the cost batching targets.

Caveats:
- Not a like-for-like comparison of resources: LitServe runs more processes than a single FastAPI worker. CPU per request (`docker stats`) was not recorded for these runs.
- The best FastAPI result (4 workers, ~1118 req/s) is still well above LitServe with 1 inference worker.
- The 395 ms max at 1 user is a one-off, probably the first request paying CUDA or library initialization. It does not affect p99, but warm up before measuring when comparing small differences.

Prediction for batching (`max_batch_size=8`, 8 users): 8× throughput is an upper bound that will not be reached, because:
- Only the forward pass is batched. HTTP parsing, the process hops, `decode_request`, `encode_response` and tokenization still run once per request (Amdahl's law).
- A batch of 8 is not free (m > 0, plus batch tokenization and padding).
- Batches will not always be full: in a closed loop the 8 requests return at slightly different times, so with a short `batch_timeout` the worker often starts a batch with fewer than 8.
- Little's law: throughput = 8 ÷ latency, and latency cannot go below one round trip (HTTP hop + one batch).

Rough estimate with assumed numbers: if a batch of 8 takes ~2 ms and the HTTP and process hops add ~1.5 ms, latency at 8 users is ~3.5 ms and throughput ≈ 8 ÷ 3.5 ms ≈ 2,300 req/s (~3.8×). Log the actual batch sizes `predict` receives, and check `docker stats` / `htop` to see whether the API server, the inference worker or Locust hits 100%.

### LitServe with batching (max batch size 8, 1 inference worker)

#### Code changes for batching
- `decode_request` (per request): stores the prompt in `context` and returns the raw prompt text.
- `batch` (per batch): tokenizes the whole list of prompts together with `padding=True` and moves the tensors to the device once.
- `predict` (per batch): one forward pass, returns a **list** with one label per input, in input order.
- `encode_response` (per request): builds `Result` from the label and the prompt in `context`.

Gotchas found along the way:
- `batch` is only called when `max_batch_size > 1`. Otherwise LitServe uses `SingleLoop`, which skips `batch` and passes the raw prompt string to `predict`. The warning about this only fires when both `batch` and `unbatch` are overridden.
- If `predict` returns a single string in batched mode, the default `unbatch` (`list(output)`) splits it into characters, e.g. `"POSITIVE"` → `['P', 'O', ...]`, which fails the "Batch size mismatch" check (or silently passes when the batch size happens to equal the string length).
- Calling `.item()` per row copies each value from the GPU separately and syncs each time. `.tolist()` copies the whole tensor in one transfer.
- `max_batch_size` and `batch_timeout` belong on the `LitAPI` constructor. Passing them to `LitServer` still works in 0.2.19 but is deprecated.
- `batch_timeout` is in **seconds**.
- Average batch size was recorded with counters on the API object (requests and batches per worker), printed every few seconds and reset, instead of printing every batch in the hot loop.

#### Results with `batch_timeout` = 1 ms

| | 1 user | 8 users | 32 users |
| :--- | :---: | :---: | :---: |
| **Request Count** | 21114 | 71995 | 133136 |
| **Requests/s** | 358~ | 1220~ | 2256~ |
| **p50 latency** | 3 | 7 | 14 |
| **p95 latency** | 3 | 8 | 16 |
| **p99 latency** | 4 | 8 | 17 |
| **Average latency** | 2.77 | 6.44 | 14.06 |
| **Average batch size** | 1 | varied 1–8 | always 8 |
| **Failures** | 0 | 0 | 0 |

Compared with LitServe without batching:

| Users | No batching req/s | Batch 8 req/s | Change | p50 / p99 (no batching → batch 8) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | ~384 | ~358 | 0.93× | 3 / 3 → 3 / 4 ms |
| 8 | ~606 | ~1220 | 2.0× | 13 / 16 → 7 / 8 ms |
| 32 | ~608 | ~2256 | 3.7× | 52 / 60 → 14 / 17 ms |

Little's law still holds: 8 / 1220 ≈ 6.6 ms (actual 6.4), 32 / 2256 ≈ 14.2 ms (actual 14.1). At 32 users this is about **twice the best FastAPI result** (4 workers, ~1118 req/s) with p99 17 ms instead of 46 ms, from a single inference worker.

**32 users: estimate of F and m.** Batches were always full (average batch size exactly 8, which can only happen if no batch was smaller). Assuming the inference worker was busy 100% of the time (not yet verified with `htop`):
```
2256 req/s ÷ 8      ≈ 282 batches/s → ~3.55 ms per batch of 8
no batching         ≈ ~1.65 ms per batch of 1

F + m  ≈ 1.65 ms
F + 8m ≈ 3.55 ms
→ m ≈ 0.27 ms, F ≈ 1.4 ms
```
A batch of 8 costs only ~2.15× a batch of 1, and the fixed cost per call is about 5× the cost per extra input, which agrees with the py-spy profile. If the worker was ever idle, the real batch time is lower, so treat 3.55 ms as an upper bound.

**8 users: why only 2×.** With 8 users and a max batch of 8 there is roughly one batch in flight at a time. The worker runs a batch (~3.5 ms), then sits idle while the responses travel back (response queue → API server → HTTP → Locust) and the next requests travel in. Latency of ~6.4 ms ≈ ~3.5 ms batch + ~3 ms round trip, so the worker is busy only about half the time. At 32 users other requests are already queued while one batch's round trip completes, so the worker stays busy (pipelining).

**8 users: why batch size varied 1–8.** The 8 requests do not come back at the same moment, since each response passes through the queues, the API server, HTTP and Locust one after another. Whatever has arrived when the batch window closes becomes a batch. Once the users split into groups, the groups finish and return at different times and keep being batched separately, so the batch size wanders.

**Prediction vs. actual.** I predicted ~2,300 req/s at 8 users. That was reached at 32 users, and 8 users got about half. The assumed numbers were off (a batch of 8 took ~3.5 ms, not ~2 ms, and the round trip was ~3 ms, not ~1.5 ms), and the estimate ignored the worker idling during the round trip of a single batch in flight.

#### `batch_timeout` = 3 ms vs. 1 ms

| | 1 ms timeout | 3 ms timeout |
| :--- | :---: | :---: |
| **1 user req/s** | ~358 | ~207 |
| **1 user p50 / p99** | 3 / 4 ms | 5 / 5 ms |
| **1 user average latency** | 2.77 ms | 4.81 ms |
| **8 users req/s** | ~1220 | ~1179 |
| **8 users p50 / p99** | 7 / 8 ms | 6 / 10 ms |
| **8 users average batch size** | varied 1–8 | varied 4–8 |

**8 users: fuller batches, same throughput.** The longer window collected more of the requests arriving at slightly different times, so batches rose from 1–8 to 4–8. Throughput did not improve (~3% lower, possibly noise with one run each). Fuller batches saved inference worker time, but the worker was not the bottleneck at 8 users since it was idle during each round trip. In a closed loop with 8 users, throughput = 8 ÷ latency, and the extra window wait roughly cancelled out the savings from paying F less often. At this concurrency, raising throughput means lowering per-cycle latency, and a longer timeout does the opposite.

**1 user: how the timeout adds latency.** From LitServe's `collate_requests` (`litserve/loops/base.py`), the batch window starts when the worker begins collecting, not when a request arrives. With 1 user in a closed loop:
1. The worker sends a response and immediately opens a new collection window.
2. The response travels back to Locust, Locust sends the next request, and it travels back to the worker. Call that time **r**.
3. The request arrives r into the window and waits for the rest of it: **timeout − r**.

Both measurements fit with **r ≈ 0.8 ms** (time a request spends outside the inference worker):
```
1 ms − 0.8 ms ≈ 0.2 ms added   (measured: 2.77 − 2.58 ≈ +0.2 ms)
3 ms − 0.8 ms ≈ 2.2 ms added   (measured: 4.81 − 2.58 ≈ +2.2 ms)
```
This is inferred from two consistent measurements, not measured directly.

**Rule behind the results:** batching raises throughput only when there is a backlog. At 32 users there is always a queue (3.7×). At 8 users there is sometimes a queue (2×), and tuning the window cannot do much more. At 1 user there is never a queue, so batching only costs latency.

**Decision:** 1 ms timeout. 3 ms is worse everywhere it differs: −46% throughput at 1 user versus no batching, and no gain at 8 users.

Note: the 8 users CSV for the 1 ms run (`8_users_litserve_batch_8_stats.csv`) was overwritten by the 3 ms run. The 1 ms numbers above are from the run output recorded before it was overwritten. Use a filename that includes the timeout for future runs.

#### Next
- `htop` during a 32-user run: is the inference worker at 100%? If yes, the cap of 8 is now the limit, so try `max_batch_size=16`, with a prediction from F ≈ 1.4 ms, m ≈ 0.27 ms first. If the API server or Locust is at 100%, that is the new ceiling.
- Optional: `batch_timeout=0` (LitServe takes whatever is already queued with no wait). Expected: same as no batching at 1 user, still full batches at 32 users, somewhat smaller batches at 8 users.
- `workers_per_device=2` at 8 users, to see whether a second worker can run one batch while the other's round trip completes.