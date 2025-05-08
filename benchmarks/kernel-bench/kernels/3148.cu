#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Helper function for maximum
template <typename scalar_t>
__device__ inline scalar_t my_max(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

// Warp-level reduction for maximum using shuffle intrinsics
template <typename scalar_t>
__device__ inline scalar_t warpReduceMax(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = my_max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum using shuffle intrinsics
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel that computes LogSoftmax using warp-level shuffle reductions
// Each block processes one row (batch element) of the input
template <typename scalar_t>
__global__ void log_softmax_warp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {
    // Identify the batch element this block is handling
    int batch = blockIdx.x;
    const scalar_t* input_row = input + batch * dim_size;
    scalar_t* output_row = output + batch * dim_size;

    // Step 1: Compute the maximum value in the row for numerical stability
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        local_max = my_max(local_max, input_row[i]);
    }
    // Intra-warp reduction to get partial maximum
    local_max = warpReduceMax<scalar_t>(local_max);

    // Use shared memory to store the max from each warp
    __shared__ scalar_t shared_max[32]; // assume at most 32 warps per block
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();

    // First warp reduces the partial maximums
    scalar_t row_max = -std::numeric_limits<scalar_t>::infinity();
    if (threadIdx.x < (blockDim.x / warpSize)) {
        row_max = shared_max[threadIdx.x];
    }
    if (threadIdx.x < (blockDim.x / warpSize)) {
        row_max = warpReduceMax<scalar_t>(row_max);
    }
    if (threadIdx.x == 0) {
        shared_max[0] = row_max;
    }
    __syncthreads();
    row_max = shared_max[0];

    // Step 2: Compute the sum of exp(input - row_max)
    scalar_t local_sum = 0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        local_sum += exp(input_row[i] - row_max);
    }
    local_sum = warpReduceSum<scalar_t>(local_sum);

    __shared__ scalar_t shared_sum[32];
    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    scalar_t row_sum = 0;
    if (threadIdx.x < (blockDim.x / warpSize)) {
        row_sum = shared_sum[threadIdx.x];
    }
    if (threadIdx.x < (blockDim.x / warpSize)) {
        row_sum = warpReduceSum<scalar_t>(row_sum);
    }
    if (threadIdx.x == 0) {
        shared_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = shared_sum[0];
    scalar_t log_sum = log(row_sum);

    // Step 3: Write the final log_softmax output
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        output_row[i] = (input_row[i] - row_max) - log_sum;
    }
}

// Host function that prepares and launches the CUDA kernel
torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input so that the reduction is performed on the last dimension
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

    // Launch one block per batch element with a fixed 256 threads per block
    int threads = 256;
    int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_warp_cuda", ([&] {
        log_softmax_warp_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA) with warp shuffle reduction");
}
