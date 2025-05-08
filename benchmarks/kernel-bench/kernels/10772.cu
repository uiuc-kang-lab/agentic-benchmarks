#include <torch/extension.h>

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    auto sizes = x.sizes().vec();
    auto strides = x.strides().vec();
    auto min_stride = *std::min_element(strides.begin(), strides.end());
    
    if (strides[dim] == min_stride) {
        auto x_flipped = x.contiguous().flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        return cumsum.flip(dim);
    }
    
    std::vector<int64_t> permute_dims(x.dim());
    std::iota(permute_dims.begin(), permute_dims.end(), 0);
    std::swap(permute_dims[dim], permute_dims.back());
    
    auto x_permuted = x.permute(permute_dims).contiguous();
    auto x_flipped = x_permuted.flip(-1);
    auto cumsum = x_flipped.cumsum(-1);
    auto cs_flipped = cumsum.flip(-1);
    auto out = cs_flipped.permute(permute_dims).contiguous();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Optimized reverse cumsum with memory coalescing");
}