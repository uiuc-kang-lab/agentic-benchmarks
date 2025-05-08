#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void log_softmax_warp_vectorized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const scalar_t* row_in = input + batch_idx * dim_size;
    scalar_t* row_out = output + batch_idx * dim_size;

    // Vectorized processing setup
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(scalar_t);
    const int vec_dim = dim_size / VEC_SIZE;
    
    // Phase 1: Find max value with vector loads
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < vec_dim; i += BLOCK_SIZE) {
        float4 vec_data = *reinterpret_cast<const float4*>(row_in + i * VEC_SIZE);
        scalar_t elements[VEC_SIZE];
        *reinterpret_cast<float4*>(elements) = vec_data;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            max_val = max(max_val, elements[v]);
        }
    }
    // Handle remaining elements
    for (int i = vec_dim * VEC_SIZE + tid; i < dim_size; i += BLOCK_SIZE) {
        max_val = max(max_val, row_in[i]);
    }

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    
    if (tid % 32 == 0)
        reinterpret_cast<volatile scalar_t*>(row_out)[tid / 32] = max_val;
    __syncthreads();

    // Block-level max reduction
    if (tid == 0) {
        scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < BLOCK_SIZE / 32; ++i)
            block_max = max(block_max, row_out[i]);
        row_out[0] = block_max;
    }
    __syncthreads();
    max_val = row_out[0];

    // Phase 2: Compute sum of exponents with vector ops
    scalar_t sum = 0;
    for (int i = tid; i < vec_dim; i += BLOCK_SIZE) {
        float4 vec_data = *reinterpret_cast<const float4*>(row_in + i * VEC_SIZE);
        scalar_t elements[VEC_SIZE];
        *reinterpret_cast<float4*>(elements) = vec_data;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v)
            sum += exp(elements[v] - max_val);
    }
    // Handle remaining elements
    for (int i = vec_dim * VEC_SIZE + tid; i < dim_size; i += BLOCK_SIZE)
        sum += exp(row_in[i] - max_val);

    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (tid % 32 == 0)
        reinterpret_cast<volatile scalar_t*>(row_out)[tid / 32] = sum;
    __syncthreads();

    // Block-level sum reduction
    if (tid == 0) {
        scalar_t block_sum = 0;
        for (int i = 0; i < BLOCK_SIZE / 32; ++i)
            block_sum += row_out[i];
        row_out[0] = log(block_sum);
    }
    __syncthreads();
    scalar_t log_sum = row_out[0];

    // Phase 3: Write final values with vector stores
    for (int i = tid; i < vec_dim; i += BLOCK_SIZE) {
        float4 vec_data = *reinterpret_cast<const float4*>(row_in + i * VEC_SIZE);
        scalar_t elements[VEC_SIZE];
        *reinterpret_cast<float4*>(elements) = vec_data;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v)
            elements[v] = elements[v] - max_val - log_sum;
        *reinterpret_cast<float4*>(row_out + i * VEC_SIZE) = *reinterpret_cast<float4*>(elements);
    }
    // Handle remaining elements
    for (int i = vec_dim * VEC_SIZE + tid; i < dim_size; i += BLOCK_SIZE)
        row_out[i] = row_in[i] - max_val - log_sum;
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");

    // Permute to last dimension
    auto dims = input.sizes().vec();
    int64_t last_dim = dim >= 0 ? dim : input.dim() + dim;
    std::swap(dims[last_dim], dims.back());
    input = input.permute(dims).contiguous();
    
    const int64_t batch_size = input.numel() / input.size(-1);
    const int64_t dim_size = input.size(-1);
    auto output = torch::empty_like(input);

    // Choose block size based on dimension size
    int block_size = 256;
    if (dim_size <= 256) block_size = 128;
    if (dim_size <= 128) block_size = 64;
    if (dim_size <= 64) block_size = 32;

    dim3 blocks(batch_size);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_cuda", ([&] {
        switch(block_size) {
            case 32: log_softmax_warp_vectorized_kernel<scalar_t, 32><<<blocks, 32>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dim_size); break;
            case 64: log_softmax_warp_vectorized_kernel<scalar_t, 64><<<blocks, 64>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dim_size); break;
            case 128: log_softmax_warp_vectorized_kernel<scalar_t, 128><<<blocks, 128>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dim_size); break;
            default: log_softmax_warp_vectorized_kernel<scalar_t, 256><<<blocks, 256>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dim_size);
        }
    }));

    // Restore original dimensions
    std::swap(dims[last_dim], dims.back());
    return output.permute(dims);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
}