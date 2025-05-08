#include <torch/extension.h>

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    auto cumsum = x.cumsum(dim);
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