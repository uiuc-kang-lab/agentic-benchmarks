#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel_2d(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    // Each block handles one batch element
    const int batch_idx = blockIdx.x;
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int thread_idx = thread_row * blockDim.x + thread_col;
    const int total_threads = blockDim.x * blockDim.y;

    // Pointers to current batch element
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Shared memory for reductions
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Initialize max_val
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Compute max value with grid-stride loop
    for (int idx = thread_idx; idx < dim_size; idx += total_threads) {
        max_val = max(max_val, input_row[idx]);
    }

    // Store in shared memory
    sdata[thread_idx] = max_val;
    __syncthreads();

    // Reduce within block using 2D structure
    for (int stride = total_threads/2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            sdata[thread_idx] = max(sdata[thread_idx], sdata[thread_idx + stride]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Compute sum of exp(input - max_val)
    scalar_t sum = 0;
    for (int idx = thread_idx; idx < dim_size; idx += total_threads) {
        scalar_t val = exp(input_row[idx] - max_val);
        output_row[idx] = val;  // Store intermediate result
        sum += val;
    }

    // Store partial sums
    sdata[thread_idx] = sum;
    __syncthreads();

    // Reduce sum within block
    for (int stride = total_threads/2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            sdata[thread_idx] += sdata[thread_idx + stride];
        }
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    scalar_t log_sum = log(sum);

    // Compute final output with grid-stride loop
    for (int idx = thread_idx; idx < dim_size; idx += total_threads) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
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

    // Use 32x32 thread block configuration
    dim3 threads(32, 32);
    const int blocks = batch_size;
    const int shared_mem_size = 1024 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel_2d<scalar_t><<<blocks, threads, shared_mem_size>>>(
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