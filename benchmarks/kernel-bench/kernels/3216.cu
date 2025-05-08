#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel uses manual loop unrolling with #pragma unroll for critical reduction loops.
// The unrolling decreases loop overhead and improves ILP, while maintaining full precision.

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_unroll(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    // Each block processes one row (batch element)
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    const unsigned int mask = 0xffffffff;

    // Step 1: Compute local maximum
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t val = input_row[i];
        local_max = (val > local_max) ? val : local_max;
    }

    // Warp-level reduction for maximum using unrolled loop
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(mask, local_max, offset);
        local_max = (temp > local_max) ? temp : local_max;
    }

    // Allocate shared memory for per-warp results
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);
    int warp_id = tid / warpSize;
    if ((tid % warpSize) == 0) {
        shared_data[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction over warps for maximum (performed by thread 0)
    if (tid == 0) {
        int num_warps = (blockSize + warpSize - 1) / warpSize;
        scalar_t global_max = shared_data[0];
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            global_max = (shared_data[i] > global_max) ? shared_data[i] : global_max;
        }
        shared_data[0] = global_max;  // broadcast global maximum
    }
    __syncthreads();
    scalar_t global_max = shared_data[0];

    // Step 2: Compute the sum of exp(val - global_max)
    scalar_t local_sum = 0;
    for (int i = tid; i < dim_size; i += blockSize) {
        scalar_t exp_val = exp(input_row[i] - global_max);
        local_sum += exp_val;
        output_row[i] = exp_val;  // store intermediate result
    }

    // Warp-level reduction for sum, unrolling the loop
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    if ((tid % warpSize) == 0) {
        shared_data[warp_id] = local_sum;
    }
    __syncthreads();

    scalar_t global_sum;
    if (tid == 0) {
        int num_warps = (blockSize + warpSize - 1) / warpSize;
        global_sum = shared_data[0];
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            global_sum += shared_data[i];
        }
        shared_data[0] = global_sum;  // broadcast global sum
    }
    __syncthreads();
    global_sum = shared_data[0];

    scalar_t log_sum = log(global_sum);

    // Step 3: Compute final log softmax output
    for (int i = tid; i < dim_size; i += blockSize) {
        output_row[i] = (input_row[i] - global_max) - log_sum;
    }
}

// Host function launching the kernel

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = (dim >= 0) ? dim : dim + ndim;

    // Permute input to bring 'dim' to the last dimension
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

    // Choose number of threads: next power of two of dim_size, capped at 1024
    int threads = 1;
    while (threads < dim_size) threads <<= 1;
    if (threads > 1024) threads = 1024;

    // Compute required shared memory: one scalar per warp
    int warpSize = 32;
    int num_warps = (threads + warpSize - 1) / warpSize;
    size_t shared_mem_size = num_warps * sizeof(float); // temporary, overridden per type below

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda_unroll", ([&] {
        shared_mem_size = num_warps * sizeof(scalar_t);
        log_softmax_forward_kernel_unroll<scalar_t><<<batch_size, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    // Inverse permutation to restore original shape
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward with loop unrolling (CUDA)");
}
