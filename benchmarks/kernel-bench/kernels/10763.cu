#include <torch/extension.h>

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    const auto size = x.size(dim);
const int blockSize = 256;
const int numBlocks = (size + blockSize - 1) / blockSize;

// Allocate shared memory for the cumulative sum
__shared__ float shared_cumsum[blockSize];

// Kernel to compute cumulative sum in shared memory
for (int block = 0; block < numBlocks; ++block) {
    int start = block * blockSize;
    int end = min(start + blockSize, size);
    float sum = 0;
    for (int i = start; i < end; ++i) {
        sum += x[i];
        shared_cumsum[i - start] = sum;
    }
    __syncthreads();
    // Write back to global memory
    for (int i = start; i < end; ++i) {
        cumsum[i] = shared_cumsum[i - start];
    }
}
    auto total = cumsum.slice(dim, -1, cumsum.size(dim), 1);
    auto zeros_shape = x.sizes().vec();
    zeros_shape[dim] = 1;
    auto zeros = at::zeros(zeros_shape, x.options());
    auto shifted_cumsum = at::cat({zeros, cumsum.narrow(dim, 0, x.size(dim)-1)}, dim);
    auto reverse_cumsum = total.sub(shifted_cumsum);
    return reverse_cumsum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Optimized reverse cumulative sum");
}