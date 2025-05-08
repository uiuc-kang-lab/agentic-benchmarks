#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Utility function to find the next highest power of two
inline int next_power_of_two(int x) {
    return 1 << (32 - __builtin_clz(x - 1));
}

// Warp-level reduction for maximum
template <typename scalar_t>
__device__ inline scalar_t warpReduceMax(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction for maximum using warp reduction
template <typename scalar_t>
__device__ inline scalar_t blockReduceMax(scalar_t val) {
    __shared__ scalar_t shared[32];  // Maximum number of warps per block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceMax(val);
    if(lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only first warp participates in final reduction
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -std::numeric_limits<scalar_t>::infinity();
    if(wid == 0) {
        val = warpReduceMax(val);
    }
    return val;
}

// Warp-level reduction for sum
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for sum
template <typename scalar_t>
__device__ inline scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    if(lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<scalar_t>(0);
    if(wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Modular device function to compute the local maximum from input_row
template <typename scalar_t>
__device__ inline scalar_t compute_local_max(const scalar_t* input_row, int dim_size) {
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        local_max = max(local_max, __ldg(input_row + i));
    }
    return local_max;
}

// Modular device function to compute the local sum of exp(input - block_max)
template <typename scalar_t>
__device__ inline scalar_t compute_local_sum(const scalar_t* input_row, int dim_size, scalar_t block_max) {
    scalar_t local_sum = 0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        local_sum += exp(__ldg(input_row + i) - block_max);
    }
    return local_sum;
}

// Modular device function to compute final log-softmax
template <typename scalar_t>
__device__ inline void compute_logsoftmax(const scalar_t* input_row, scalar_t* output_row,
                                             int dim_size, scalar_t block_max, scalar_t log_sum) {
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        output_row[i] = (__ldg(input_row + i) - block_max) - log_sum;
    }
}

// Main kernel using modular device functions
template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    // Each block processes one batch element (one row)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Compute the maximum value in the row
    scalar_t local_max = compute_local_max(input_row, dim_size);
    scalar_t block_max = blockReduceMax(local_max);

    // Compute the sum of exponentials in the row
    scalar_t local_sum = compute_local_sum(input_row, dim_size, block_max);
    scalar_t block_sum = blockReduceSum(local_sum);

    scalar_t log_sum = log(block_sum);

    // Write the final log-softmax output
    compute_logsoftmax(input_row, output_row, dim_size, block_max, log_sum);
}

// Interface function callable from PyTorch
torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input so that target dimension is last
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

    // Determine block size, ensuring it is a multiple of warpsize
    int threads = next_power_of_two(dim_size);
    threads = (threads + 31) / 32 * 32;
    threads = threads < 1024 ? threads : 1024;

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        size_t shared_mem_size = 0; // Not needed as we use warp intrinsics
        log_softmax_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    // Inverse permute to restore the original tensor shape
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
}
