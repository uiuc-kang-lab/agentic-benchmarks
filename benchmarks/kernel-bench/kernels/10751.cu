#include <torch/extension.h>

#define UNROLL_FACTOR 4

template<typename scalar_t>
__global__ void reverse_cumsum_kernel(scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     int64_t dim_size,
                                     int64_t stride,
                                     int64_t num_slices) {
    const int64_t slice_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (slice_idx >= num_slices) return;

    scalar_t* slice_in = input + slice_idx * stride;
    scalar_t* slice_out = output + slice_idx * stride;
    scalar_t acc = 0;

    // Process elements in reverse order with unrolling
    int i = dim_size - 1;
    for (; i >= UNROLL_FACTOR; i -= UNROLL_FACTOR) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
            acc += slice_in[i - j];
            slice_out[i - j] = acc;
        }
    }

    // Handle remaining elements
    for (; i >= 0; --i) {
        acc += slice_in[i];
        slice_out[i] = acc;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    auto sizes = x.sizes().vec();
    int64_t dim_size = sizes[dim];
    sizes[dim] = 1;
    int64_t num_slices = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int64_t>());
    int64_t stride = x.stride(dim);
    
    auto output = torch::empty_like(x);
    
    dim3 blocks((num_slices + 255) / 256);
    dim3 threads(256);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            stride,
            num_slices
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with loop unrolling");
}