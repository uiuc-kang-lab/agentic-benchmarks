#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to compute cumulative product for a single sequence
// Using warp-level primitives for intra-warp communication to optimize small reductions

template <typename scalar_t>
__device__ scalar_t warp_prefix_prod(scalar_t val) {
    const unsigned int mask = 0xFFFFFFFF;
    int lane = threadIdx.x & 31;
    if constexpr (std::is_same<scalar_t, c10::Half>::value) {
        __half h_val = static_cast<__half>(val);
        for (int offset = 1; offset < 32; offset <<= 1) {
            __half y = __shfl_up_sync(mask, h_val, offset, 32);
            if (lane >= offset) {
                h_val = __hmul(h_val, y);
            }
        }
        return scalar_t(h_val);
    } else {
        for (int offset = 1; offset < 32; offset <<= 1) {
            scalar_t y = __shfl_up_sync(mask, val, offset, 32);
            if (lane >= offset) {
                val *= y;
            }
        }
        return val;
    }
}

// Kernel function that utilizes warp-level primitives for cumulative product computation

template <typename scalar_t>
__global__ void cumprod_kernel_warp_level(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;

    if (idx < numel / dim_size) {
        const int64_t base_offset = batch_idx * (stride * dim_size) + in_idx;
        scalar_t product = 1;
        for (int i = 0; i < dim_size; i++) {
            const int64_t offset = base_offset + i * stride;
            scalar_t val = input[offset];
            scalar_t prefix_prod = warp_prefix_prod(val);
            if (threadIdx.x % 32 == 31) // Last thread in the warp to write
                product *= prefix_prod;
            val = product;
            output[offset] = val;
        }
    }
}

torch::Tensor cumprod_cuda_forward_warp_level(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    // Get tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();

    // Calculate dimension properties
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();

    // Calculate total number of elements to process
    int64_t total_threads = numel / dim_size;

    // CUDA kernel launch parameters with increased block size
    const int threads = 512;
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_warp_level", ([&] {
        cumprod_kernel_warp_level<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            numel,
            dim_size,
            stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_warp_level, "Cumulative product forward with warp-level primitives (CUDA)");
}