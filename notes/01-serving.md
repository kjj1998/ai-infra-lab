# Notes

## Different ways of calling top-level torch functions in Pytorch
Pytorch provides different ways of calling top-level torch functions like `softmax` and `argmax`.
It can be called as a standard PyTorch function via `torch.softmax(logits)` by passing the tensor as the first argument
or called directly on a tensor object as `logits.softmax()` for e.g.

Both perform the exact same mathematical operation and return identical results.

Pytorch maps many of its top-level torch functions like `.softmax()` and `argmax()` directly onto the 
`torch.Tensor` class as methods. This allows easy chaining of operations together.

## `pip` vs `uv` for Docker image building
`uv` Python Dockerfiles are significantly faster than traditional `pip` Dockerfiles due at resolving and downloading
dependencies due to being written in Rust, utilizing aggressive parallel execution and uses an advanced dependency solver.

## Docker image optimization
Before Optimization:
- Content Size: `3.14GB`
After Optimization (multistage build):
- Content Size: `3.12GB`

Expected given that CUDA-enabled torch packages are installed