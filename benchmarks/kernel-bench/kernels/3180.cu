#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void shared_mem_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for input data and intermediate results
    extern __shared__ unsigned char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_buffer = shared_input + BLOCK_SIZE;
    
    // First phase: Find maximum value using shared memory
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    
    // Load data into shared memory in chunks
    const int num_chunks = (dim_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int idx = chunk * BLOCK_SIZE + threadIdx.x;
        
        // Load chunk into shared memory
        if (idx < dim_size) {
            shared_input[threadIdx.x] = input_row[idx];
        }
        __syncthreads();
        
        // Process chunk from shared memory
        if (idx < dim_size) {
            thread_max = max(thread_max, shared_input[threadIdx.x]);
        }
        __syncthreads();
    }

    // Warp-level reduction for maximum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // Store warp results in shared memory
    if (threadIdx.x % warpSize == 0) {
        shared_buffer[threadIdx.x / warpSize] = thread_max;
    }
    __syncthreads();

    // Final reduction for maximum (single warp)
    if (threadIdx.x < (BLOCK_SIZE / warpSize)) {
        thread_max = shared_buffer[threadIdx.x];
        #pragma unroll
        for (int offset = (BLOCK_SIZE / warpSize) / 2; offset > 0; offset /= 2) {
            thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (threadIdx.x == 0) {
            shared_buffer[0] = thread_max;
        }
    }
    __syncthreads();
    const scalar_t max_val = shared_buffer[0];

    // Second phase: Compute sum of exponentials using shared memory
    scalar_t thread_sum = 0;
    
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int idx = chunk * BLOCK_SIZE + threadIdx.x;
        
        // Load chunk into shared memory
        if (idx < dim_size) {
            shared_input[threadIdx.x] = input_row[idx];
        }
        __syncthreads();
        
        // Process chunk from shared memory
        if (idx < dim_size) {
            thread_sum += exp(shared_input[threadIdx.x] - max_val);
        }
        __syncthreads();
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        shared_buffer[threadIdx.x / warpSize] = thread_sum;
    }
    __syncthreads();

    // Final reduction for sum (single warp)
    if (threadIdx.x < (BLOCK_SIZE / warpSize)) {
        thread_sum = shared_buffer[threadIdx.x];
        #pragma unroll
        for (int offset = (BLOCK_SIZE / warpSize) / 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (threadIdx.x == 0) {
            shared_buffer[0] = thread_sum;
        }
    }
    __syncthreads();
    
    const scalar_t sum = shared_buffer[0];
    const scalar_t log_sum = log(sum);

    // Final phase: Compute output values using shared memory
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int idx = chunk * BLOCK_SIZE + threadIdx.x;
        
        // Load chunk into shared memory
        if (idx < dim_size) {
            shared_input[threadIdx.x] = input_row[idx];
        }
        __syncthreads();
        
        // Compute final values from shared memory
        if (idx < dim_size) {
            output_row[idx] = (shared_input[threadIdx.x] - max_val) - log_sum;
        }
        __syncthreads();
    }
}

torch::Tensor shared_mem_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
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
    
    // Shared memory size: input buffer + reduction buffer
    const size_t shared_mem_size = BLOCK_SIZE * sizeof(scalar_t) * 2;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_mem_logsoftmax_cuda_forward", ([&] {
        shared_mem_logsoftmax_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE, shared_mem_size>>>(
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
    m.def("forward", &shared_mem_logsoftmax_cuda_forward, "Shared Memory LogSoftmax forward (CUDA)");
}