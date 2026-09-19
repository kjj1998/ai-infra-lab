# Notes

## Different ways of calling top-level torch functions in Pytorch
Pytorch provides different ways of calling top-level torch functions like `softmax` and `argmax`.
It can be called as a standard PyTorch function via `torch.softmax(logits)` by passing the tensor as the first argument
or called directly on a tensor object as `logits.softmax()` for e.g.

Both perform the exact same mathematical operation and return identical results.

Pytorch maps many of its top-level torch functions like `.softmax()` and `argmax()` directly onto the 
`torch.Tensor` class as methods. This allows easy chaining of operations together.

