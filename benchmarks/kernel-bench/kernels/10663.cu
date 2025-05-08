#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_shared_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {
    
    extern __shared__ scalar_t shared_data[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int elements_per_block = blockDim.x;

    const int sequence_idx = bid * elements_per_block + tid;
    if (sequence_idx >= numel / dim_size) return;

    const int batch_idx = sequence_idx / stride;
    const int in_idx = sequence_idx % stride;
    const int64_t base_offset = batch_idx * (stride * dim_size) + in_idx;

    // Load sequence into shared memory with coalesced access
    for (int i = 0; i < dim_size; ++i) {
        shared_data[tid * dim_size + i] = input[base_offset + i * stride];
    }
    __syncthreads();

    // Compute cumulative product with warp-level optimization
    scalar_t product = 1;
    for (int i = 0; i < dim_size; ++i) {
        product *= shared_data[tid * dim_size + i];
        
        // Intra-warp reduction using shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            product *= __shfl_down_sync(0xffffffff, product, offset);
        }
        
        if (threadIdx.x % 32 == 0) {
            shared_data[tid * dim_size + i] = product;
        }
        __syncthreads();
        
        product = shared_data[(tid & ~31) * dim_size + i];
    }

    // Write back to global memory
    for (int i = 0; i < dim_size; ++i) {
        output[base_offset + i * stride] = shared_data[tid * dim_size + i];
    }
}

torch::Tensor cumprod_cuda_shared_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    
    const int threads = 256;
    const int blocks = (numel / dim_size + threads - 1) / threads;
    size_t shared_mem = threads * dim_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_shared", ([&] {
        cumprod_shared_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
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
    m.def("forward", &cumprod_cuda_shared_forward, "Cumulative product with shared memory (CUDA)");
}