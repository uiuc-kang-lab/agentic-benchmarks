#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_minsync(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int tid = threadIdx.x;
    const int wid = tid / warpSize;
    const int lane = tid % warpSize;
    const int batch_idx = blockIdx.x;
    const int blockSize = blockDim.x;
    const unsigned int mask = 0xffffffff;

    // Input/output pointers for current batch
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for inter-warp communication only
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* warp_results = reinterpret_cast<scalar_t*>(smem);

    // Step 1: Find max value using warp-level reduction
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockSize) {
        thread_max = max(thread_max, input_row[i]);
    }

    // Warp-level reduction without sync
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(mask, thread_max, offset);
        thread_max = max(thread_max, other);
    }

    // Only first thread in each warp writes to shared memory
    if (lane == 0) {
        warp_results[wid] = thread_max;
    }

    // Single sync point for warp results
    __syncthreads();

    // First warp reduces across all warps
    scalar_t global_max;
    if (wid == 0 && lane < (blockSize + warpSize - 1)/warpSize) {
        scalar_t warp_max = warp_results[lane];
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            scalar_t other = __shfl_down_sync(mask, warp_max, offset);
            warp_max = max(warp_max, other);
        }
        if (lane == 0) {
            warp_results[0] = warp_max;
        }
    }

    // No sync needed here as we only need first warp's result
    global_max = warp_results[0];

    // Step 2: Compute exp sum using warp-level reduction
    scalar_t thread_sum = 0;
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t val = exp(input_row[i] - global_max);
        output_row[i] = val;  // Store for later use
        thread_sum += val;
    }

    // Warp-level sum reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }

    // Single sync point for warp sums
    __syncthreads();

    // First warp reduces sums
    scalar_t global_sum;
    if (wid == 0 && lane < (blockSize + warpSize - 1)/warpSize) {
        scalar_t warp_sum = warp_results[lane];
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }
        if (lane == 0) {
            warp_results[0] = warp_sum;
        }
    }

    // No sync needed as we only need first warp's result
    global_sum = warp_results[0];
    scalar_t log_sum = log(global_sum);

    // Step 3: Compute final output
    for (int i = tid; i < dim_size; i += blockSize) {
        output_row[i] = input_row[i] - global_max - log_sum;
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

    const int threads = std::min(1024, next_power_of_two(dim_size));
    const int warps_per_block = (threads + 31) / 32;
    const size_t shared_mem_size = warps_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel_minsync<scalar_t><<<batch_size, threads, shared_mem_size>>>(
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
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
}