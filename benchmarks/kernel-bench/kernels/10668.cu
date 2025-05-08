#include <torch/extension.h>

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    // Flip the tensor along the specified dimension
    auto x_flipped = x.flip(dim);

    // Compute the cumulative sum along the same dimension
    auto cumsum = x_flipped.cumsum(dim);

    // Flip the result back to the original orientation
    auto out = cumsum.flip(dim);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum along a specified dimension (CUDA)");
}