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

### GPU (ACCELERATOR=cuda, --gpus all)

Every run is 1 minute with the exact number of users spawned with no ramp up

| ID  | Batch        | Timeout | Workers           | API | Threads | Users | Purpose                          |
|-----|--------------|---------|-------------------|-----|---------|-------|----------------------------------|
| G0a | 1            | 0       | 1                 | 1   | default | 1     | Baseline vs FastAPI GPU          |
| G0b | 1            | 0       | 1                 | 1   | default | 8     |                                  |
| G0c | 1            | 0       | 1                 | 1   | default | 32    |                                  |
| G3a | 8            | 0.001   | 1                 | 1   | default | 32    | Batching sweep                   |
| G3b | 8            | 0.005   | 1                 | 1   | default | 32    |                                  |
| G3c | 8            | 0.01    | 1                 | 1   | default | 32    |                                  |
| G3d | 16           | 0.001   | 1                 | 1   | default | 32    |                                  |
| G3e | 16           | 0.005   | 1                 | 1   | default | 32    |                                  |
| G3f | 16           | 0.01    | 1                 | 1   | default | 32    |                                  |
| G3g | 32           | 0.001   | 1                 | 1   | default | 32    |                                  |
| G3h | 32           | 0.005   | 1                 | 1   | default | 32    |                                  |
| G3i | 32           | 0.01    | 1                 | 1   | default | 32    |                                  |
| G3j | best G3      | best G3 | 1                 | 1   | default | 1     | Cost of the batching timeout     |
| G1a | 1            | 0       | 2                 | 1   | default | 32    | Inference processes only         |
| G1b | 1            | 0       | 4                 | 1   | default | 32    |                                  |
| G1c | 1            | 0       | 1                 | 2   | default | 32    | HTTP processes only              |
| G1d | 1            | 0       | 1                 | 4   | default | 32    |                                  |
| G1e | 1            | 0       | 2                 | 2   | default | 32    | Both                             |
| G1f | 1            | 0       | 4                 | 4   | default | 32    | Closest to FastAPI 4 workers     |
| G2a | 1            | 0       | best 4-process G1 |     | 2       | 32    | Reproduce thread null result     |
| G4a | best G3      | best G3 | top G1 #1         |     | default | 32    | Batching combined with processes |
| G4b | best G3      | best G3 | top G1 #2         |     | default | 32    |                                  |
| G4c | best G3      | best G3 | top G1 #3         |     | default | 32    |                                  |
| G5a | best overall |         |                   |     |         | 64    | Final table                      |
| G5b | best overall |         |                   |     |         | 128   |                                  |

#### Baseline (G0a - G0c)

|                   | 1 worker | 8 workers | 32 workers |
| :---------------- | :------: | :-------: | :--------: |
| **Request Count** |  22971   |   36386   |   36437    |
| **Requests/s**    |   389~   |   616~    |    617~    |
| **p50 latency**   |    3     |    13     |     52     |
| **p95 latency**   |    3     |    14     |     54     |
| **p99 latency**   |    3     |    15     |     59     |

#### With batching (G3a - G3i)

Batch size 8 and 32 users

| Batch Timeout (s) | 0.001  | 0.005  |  0.01  |
| :---------------- | :----: | :----: | :----: |
| **Request Count** | 135630 | 139966 | 135303 |
| **Requests/s**    | 2298~  | 2296~  | 2292~  |
| **p50 latency**   |   14   |   14   |   14   |
| **p95 latency**   |   16   |   16   |   16   |
| **p99 latency**   |   17   |   17   |   17   |

Batch size 16 and 32 users

| Batch Timeout (s) | 0.001  | 0.005  |  0.01  |
| :---------------- | :----: | :----: | :----: |
| **Request Count** | 138938 | 135552 | 140158 |
| **Requests/s**    | 2354~  | 2371~  | 2375~  |
| **p50 latency**   |   13   |   13   |   13   |
| **p95 latency**   |   15   |   15   |   15   |
| **p99 latency**   |   17   |   17   |   17   |

Batch size 32 and 32 users

| Batch Timeout (s) | 0.001  | 0.005  |  0.01  |
| :---------------- | :----: | :----: | :----: |
| **Request Count** | 138643 | 124211 | 104165 |
| **Requests/s**    | 2349~  | 2104~  | 1765~  |
| **p50 latency**   |   13   |   15   |   18   |
| **p95 latency**   |   15   |   17   |   22   |
| **p99 latency**   |   17   |   19   |   23   |

#### With different workers (G1a - G1b)

batch size 1, batch timeout 0.0s and 32 users

| Workers           |   2   |   4   |
| :---------------- | :---: | :---: |
| **Request Count** | 47973 | 56534 |
| **Requests/s**    | 813~  | 958~  |
| **p50 latency**   |  39   |  33   |
| **p95 latency**   |  40   |  35   |
| **p99 latency**   |  41   |  35   |

#### With different API servers (G1c - G1d)

batch size 1, batch timeout 0.0s, 1 worker and 32 users

| API servers       |   2   |   4   |
| :---------------- | :---: | :---: |
| **Request Count** | 36232 | 35456 |
| **Requests/s**    | 615~  | 601~  |
| **p50 latency**   |  51   |  53   |
| **p95 latency**   |  55   |  56   |
| **p99 latency**   |  58   |  58   |

#### With different workers and API servers (G1e - G1f)

batch size 1, batch timeout 0.0s and 32 users
same number of api servers and workers

| API servers / Workers |   2   |   4   |
| :-------------------- | :---: | :---: |
| **Request Count**     | 47995 | 60222 |
| **Requests/s**        | 813~  | 1020~ |
| **p50 latency**       |  39   |  33   |
| **p95 latency**       |  40   |  35   |
| **p99 latency**       |  41   |  36   |

#### With best batching and timeout from G3 and different workers (G4a - G4c)

Batch size 16, batch timeout 0.01s, 1 API server, 32 users,

| Workers           |   1    |   2    |   4    |
| :---------------- | :----: | :----: | :----: |
| **Request Count** | 139569 | 131808 | 129808 |
| **Requests/s**    | 2365~  | 2233~  | 2199~  |
| **p50 latency**   |   13   |   14   |   14   |
| **p95 latency**   |   15   |   17   |   18   |
| **p99 latency**   |   17   |   19   |   20   |

#### Best overall settings (G5a - G5b)

batch size 32, batch timeout 0.001s, 1 worker, 2 API Servers

| Users             |   64   |  128   |
| :---------------- | :----: | :----: |
| **Request Count** | 173495 | 179370 |
| **Requests/s**    | 2940~  | 3039~  |
| **p50 latency**   |   22   |   42   |
| **p95 latency**   |   25   |   46   |
| **p99 latency**   |   26   |   47   |
