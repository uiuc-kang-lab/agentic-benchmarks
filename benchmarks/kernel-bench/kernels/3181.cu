#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void coalesced_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    
    // Align input/output pointers to row start
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for reductions
    __shared__ scalar_t smem[BLOCK_SIZE];
    __shared__ scalar_t warp_max[BLOCK_SIZE/warpSize];
    __shared__ scalar_t warp_sum[BLOCK_SIZE/warpSize];

    // Phase 1: Coalesced max finding using vectorized loads
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    
    // Vector size (4 floats = 128 bits)
    constexpr int vec_size = 4;
    const int vec_elements = dim_size / vec_size;
    const float4* input_vec = reinterpret_cast<const float4*>(input_row);
    
    // Ensure coalesced access by having consecutive threads read consecutive vectors
    #pragma unroll 4
    for (int base_idx = tid; base_idx < vec_elements; base_idx += BLOCK_SIZE) {
        float4 vec_val = input_vec[base_idx];
        thread_max = max(thread_max, max(max(vec_val.x, vec_val.y), 
                                       max(vec_val.z, vec_val.w)));
    }
    
    // Handle remaining elements with coalesced access
    const int remainder_start = vec_elements * vec_size;
    for (int idx = remainder_start + tid; idx < dim_size; idx += BLOCK_SIZE) {
        thread_max = max(thread_max, input_row[idx]);
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // First warp reduces warp results
    if (tid < (BLOCK_SIZE/warpSize)) {
        scalar_t max_val = warp_max[tid];
        for (int i = tid + BLOCK_SIZE/warpSize; i < BLOCK_SIZE/warpSize; i += BLOCK_SIZE/warpSize) {
            max_val = max(max_val, warp_max[i]);
        }
        warp_max[tid] = max_val;
    }
    __syncthreads();

    const scalar_t final_max = warp_max[0];

    // Phase 2: Coalesced sum computation using vectorized loads
    scalar_t thread_sum = 0;
    
    #pragma unroll 4
    for (int base_idx = tid; base_idx < vec_elements; base_idx += BLOCK_SIZE) {
        float4 vec_val = input_vec[base_idx];
        thread_sum += exp(vec_val.x - final_max);
        thread_sum += exp(vec_val.y - final_max);
        thread_sum += exp(vec_val.z - final_max);
        thread_sum += exp(vec_val.w - final_max);
    }

    for (int idx = remainder_start + tid; idx < dim_size; idx += BLOCK_SIZE) {
        thread_sum += exp(input_row[idx] - final_max);
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane_id == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces partial sums
    if (tid < (BLOCK_SIZE/warpSize)) {
        scalar_t sum = warp_sum[tid];
        for (int i = tid + BLOCK_SIZE/warpSize; i < BLOCK_SIZE/warpSize; i += BLOCK_SIZE/warpSize) {
            sum += warp_sum[i];
        }
        warp_sum[tid] = sum;
    }
    __syncthreads();

    const scalar_t final_sum = warp_sum[0];
    const scalar_t log_sum = log(final_sum);

    // Phase 3: Coalesced output writing using vectorized stores
    float4* output_vec = reinterpret_cast<float4*>(output_row);
    
    #pragma unroll 4
    for (int base_idx = tid; base_idx < vec_elements; base_idx += BLOCK_SIZE) {
        float4 vec_val = input_vec[base_idx];
        float4 result;
        result.x = (vec_val.x - final_max) - log_sum;
        result.y = (vec_val.y - final_max) - log_sum;
        result.z = (vec_val.z - final_max) - log_sum;
        result.w = (vec_val.w - final_max) - log_sum;
        output_vec[base_idx] = result;
    }

    // Handle remaining elements with coalesced access
    for (int idx = remainder_start + tid; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (input_row[idx] - final_max) - log_sum;
    }
}

torch::Tensor coalesced_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 || 
               input.scalar_type() == torch::kFloat64,
               "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) permute_dims.push_back(i);
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    // Choose block size to maximize coalescing
    const int BLOCK_SIZE = 256;
    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_logsoftmax_forward_cuda", ([&] {
        coalesced_logsoftmax_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
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
    m.def("forward", &coalesced_logsoftmax_cuda_forward, "Coalesced LogSoftmax forward (CUDA)");
}