#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val, const unsigned int mask = 0xffffffff) {
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(mask, val, offset);
        val = max(val, temp);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val, const unsigned int mask = 0xffffffff) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t block_reduce_max(scalar_t val, scalar_t* shared_data, const int tid) {
    const int warpSize = 32;
    const int warp_id = tid / warpSize;
    
    // First reduce within warps
    val = warp_reduce_max(val);
    
    // Write reduced warp values to shared memory
    if (tid % warpSize == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();
    
    // First thread reduces across warps
    if (tid == 0) {
        scalar_t max_val = shared_data[0];
        #pragma unroll
        for (int i = 1; i < blockDim.x / warpSize; i++) {
            max_val = max(max_val, shared_data[i]);
        }
        shared_data[0] = max_val;
    }
    __syncthreads();
    
    return shared_data[0];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t val, scalar_t* shared_data, const int tid) {
    const int warpSize = 32;
    const int warp_id = tid / warpSize;
    
    // First reduce within warps
    val = warp_reduce_sum(val);
    
    // Write reduced warp values to shared memory
    if (tid % warpSize == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();
    
    // First thread reduces across warps
    if (tid == 0) {
        scalar_t sum = shared_data[0];
        #pragma unroll
        for (int i = 1; i < blockDim.x / warpSize; i++) {
            sum += shared_data[i];
        }
        shared_data[0] = sum;
    }
    __syncthreads();
    
    return shared_data[0];
}

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_modular(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;
    
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);
    
    // Find maximum value
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockDim.x) {
        local_max = max(local_max, input_row[i]);
    }
    
    scalar_t max_val = block_reduce_max(local_max, shared_data, tid);
    
    // Compute exp sum
    scalar_t local_sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        scalar_t exp_val = exp(input_row[i] - max_val);
        output_row[i] = exp_val;
        local_sum += exp_val;
    }
    
    scalar_t sum_val = block_reduce_sum(local_sum, shared_data, tid);
    scalar_t log_sum = log(sum_val);
    
    // Compute final output
    for (int i = tid; i < dim_size; i += blockDim.x) {
        output_row[i] = input_row[i] - max_val - log_sum;
    }
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) permute_dims.push_back(i);
    }
    permute_dims.push_back(dim);
    
    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);
    auto output = torch::empty_like(input);

    // Round up threads to nearest warp size (32) while staying within 1024 limit
const int threads = std::min(1024, ((static_cast<int>(dim_size) + 31) / 32) * 32);
    const int warps_per_block = (threads + 31) / 32;
    const size_t shared_mem_size = warps_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_modular", ([&] {
        log_softmax_forward_kernel_modular<scalar_t><<<batch_size, threads, shared_mem_size>>>(
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
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward modular (CUDA)");
}