#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Warp-synchronous max
template <typename scalar_t>
__inline__ __device__ scalar_t warp_reduce_max(scalar_t max_val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    return max_val;
}

// Warp-synchronous sum
template <typename scalar_t>
__inline__ __device__ scalar_t warp_reduce_sum(scalar_t sum) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

// Kernel function for computing log softmax
template <typename scalar_t>
__global__ void log_softmax_warp_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    // Each block handles one batch element (row)
    int batch_idx = blockIdx.x;

    // Pointers to the input/output row
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Initialize max_val
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    // Compute max value
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        scalar_t val = input_row[idx];
        max_val = max(max_val, val);
    }
    
    // Warp-reduce max
    max_val = warp_reduce_max(max_val);
    
    // Broadcast within warp
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Compute sum of exp(input - max_val)
    scalar_t sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        scalar_t val = exp(input_row[idx] - max_val);
        output_row[idx] = val;  // Save for reuse
        sum += val;
    }

    // Warp-reduce sum
    sum = warp_reduce_sum(sum);

    // Broadcast sum within warp
    sum = __shfl_sync(0xffffffff, sum, 0);

    scalar_t log_sum = log(sum);

    // Compute output
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");

    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input to bring dim to the last dimension
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

    // Choose threads as the next highest power of 2 of dim_size, limited to 1024
    int threads = next_power_of_two(dim_size);
    threads = threads < 1024 ? threads : 1024;

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_warp_optimized", ([&] {
        log_softmax_warp_optimized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    // Inverse permute to restore original shape
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
