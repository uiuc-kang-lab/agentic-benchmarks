#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void strided_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for reductions
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    // Phase 1: Find maximum using vectorized loads when possible
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    
    // Vector loading for aligned portions
    const int vec_size = 4;
    const int vec_elements = dim_size / vec_size;
    const float4* input_vec = reinterpret_cast<const float4*>(input_row);
    
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < vec_elements; idx += BLOCK_SIZE) {
        float4 vec_val = input_vec[idx];
        thread_max = max(thread_max, max(max(vec_val.x, vec_val.y), max(vec_val.z, vec_val.w)));
    }
    
    // Handle remaining elements
    const int remainder_start = vec_elements * vec_size;
    for (int idx = remainder_start + threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        thread_max = max(thread_max, input_row[idx]);
    }

    // Warp reduction for maximum
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(mask, thread_max, offset));
    }

    // Block reduction for maximum
    if (threadIdx.x % warpSize == 0) {
        shared_data[threadIdx.x / warpSize] = thread_max;
    }
    __syncthreads();

    if (threadIdx.x < (BLOCK_SIZE / warpSize)) {
        thread_max = shared_data[threadIdx.x];
    }
    
    if (threadIdx.x == 0) {
        scalar_t block_max = thread_max;
        for (int i = 1; i < (BLOCK_SIZE / warpSize); i++) {
            block_max = max(block_max, shared_data[i]);
        }
        shared_data[0] = block_max;
    }
    __syncthreads();
    
    const scalar_t max_val = shared_data[0];

    // Phase 2: Compute sum of exponentials using vectorized operations
    scalar_t thread_sum = 0;
    
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < vec_elements; idx += BLOCK_SIZE) {
        float4 vec_val = input_vec[idx];
        thread_sum += exp(vec_val.x - max_val);
        thread_sum += exp(vec_val.y - max_val);
        thread_sum += exp(vec_val.z - max_val);
        thread_sum += exp(vec_val.w - max_val);
    }
    
    for (int idx = remainder_start + threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        thread_sum += exp(input_row[idx] - max_val);
    }

    // Warp reduction for sum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        shared_data[threadIdx.x / warpSize] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t total_sum = 0;
        for (int i = 0; i < (BLOCK_SIZE / warpSize); i++) {
            total_sum += shared_data[i];
        }
        shared_data[0] = log(total_sum);
    }
    __syncthreads();
    
    const scalar_t log_sum = shared_data[0];

    // Phase 3: Compute final values with vectorized stores when possible
    float4* output_vec = reinterpret_cast<float4*>(output_row);
    
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < vec_elements; idx += BLOCK_SIZE) {
        float4 vec_val = input_vec[idx];
        float4 result;
        result.x = (vec_val.x - max_val) - log_sum;
        result.y = (vec_val.y - max_val) - log_sum;
        result.z = (vec_val.z - max_val) - log_sum;
        result.w = (vec_val.w - max_val) - log_sum;
        output_vec[idx] = result;
    }
    
    for (int idx = remainder_start + threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

torch::Tensor strided_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            permute_dims.push_back(i);
        }
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    constexpr int BLOCK_SIZE = 256;
    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "strided_logsoftmax_cuda_forward", ([&] {
        strided_logsoftmax_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &strided_logsoftmax_cuda_forward, "Strided LogSoftmax forward (CUDA)");
}