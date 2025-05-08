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

    // Fixed block size of 256 threads for optimal occupancy
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Shared memory declaration with proper alignment
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Input/output row pointers
    const scalar_t* input_row = input + bid * dim_size;
    scalar_t* output_row = output + bid * dim_size;

    // Find max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Grid-stride loop for max computation
    for (int idx = tid; idx < dim_size; idx += 256) {
        max_val = max(max_val, __ldg(input_row + idx));
    }

    // Warp-level reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    // First thread in each warp writes to shared memory
    if (tid % 32 == 0) {
        sdata[tid / 32] = max_val;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < 8) {
        max_val = max(sdata[tid], sdata[tid + 8]);
        if (tid < 4) max_val = max(max_val, sdata[tid + 4]);
        if (tid < 2) max_val = max(max_val, sdata[tid + 2]);
        if (tid < 1) max_val = max(max_val, sdata[tid + 1]);
    }
    __syncthreads();

    max_val = (tid == 0) ? sdata[0] : max_val;
    __syncthreads();

    // Compute sum of exponentials
    scalar_t sum = 0;
    
    // Grid-stride loop for sum computation
    for (int idx = tid; idx < dim_size; idx += 256) {
        scalar_t val = exp(__ldg(input_row + idx) - max_val);
        output_row[idx] = val;
        sum += val;
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid % 32 == 0) {
        sdata[tid / 32] = sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < 8) {
        sum = sdata[tid] + sdata[tid + 8];
        if (tid < 4) sum += sdata[tid + 4];
        if (tid < 2) sum += sdata[tid + 2];
        if (tid < 1) sum += sdata[tid + 1];
    }
    __syncthreads();

    scalar_t log_sum = log(sdata[0]);

    // Compute final output
    for (int idx = tid; idx < dim_size; idx += 256) {
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

    // Fixed block size of 256 threads
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        size_t shared_mem_size = (threads / 32) * sizeof(scalar_t);
        log_softmax_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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