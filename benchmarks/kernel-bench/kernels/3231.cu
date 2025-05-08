#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Fixed block size of 128 threads optimized for H100
    constexpr int BLOCK_SIZE = 128;
    
    const scalar_t* __restrict__ input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Static shared memory allocation
    __shared__ scalar_t sdata[BLOCK_SIZE];

    // Initialize max_val
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Compute max value using grid-stride loop
    #pragma unroll 4
    for (int idx = tid; idx < dim_size; idx += BLOCK_SIZE) {
        max_val = max(max_val, __ldg(input_row + idx));
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Reduce max value within block
    #pragma unroll
    for (int offset = BLOCK_SIZE/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = max(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Compute sum of exp(input - max_val)
    scalar_t sum = 0;
    #pragma unroll 4
    for (int idx = tid; idx < dim_size; idx += BLOCK_SIZE) {
        scalar_t val = exp(__ldg(input_row + idx) - max_val);
        output_row[idx] = val;
        sum += val;
    }

    sdata[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int offset = BLOCK_SIZE/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    scalar_t log_sum = log(sum);

    // Compute final output
    #pragma unroll 4
    for (int idx = tid; idx < dim_size; idx += BLOCK_SIZE) {
        output_row[idx] = (__ldg(input_row + idx) - max_val) - log_sum;
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

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) permute_dims.push_back(i);
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    constexpr int BLOCK_SIZE = 128;
    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
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