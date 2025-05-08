#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

inline int next_power_of_two(int x) {
    return 1 << (32 - __builtin_clz(x - 1));
}

// Warp reduction utility function
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const scalar_t* __restrict__ input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Find max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    #pragma unroll 4
    for (int idx = tid; idx < dim_size; idx += blockDim.x) {
        max_val = max(max_val, __ldg(input_row + idx));
    }
    max_val = warp_reduce_max(max_val);

    if (tid % 32 == 0) {
        sdata[tid / 32] = max_val;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        max_val = sdata[tid];
        max_val = warp_reduce_max(max_val);
        if (tid == 0) {
            sdata[0] = max_val;
        }
    }
    __syncthreads();
    max_val = sdata[0];

    // Compute exp sum
    scalar_t sum = 0;
    #pragma unroll 4
    for (int idx = tid; idx < dim_size; idx += blockDim.x) {
        sum += exp(__ldg(input_row + idx) - max_val);
    }
    sum = warp_reduce_sum(sum);

    if (tid % 32 == 0) {
        sdata[tid / 32] = sum;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        sum = sdata[tid];
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            sdata[0] = sum;
        }
    }
    __syncthreads();
    sum = sdata[0];
    scalar_t log_sum = log(sum);

    // Final output computation
    #pragma unroll 4
    for (int idx = tid; idx < dim_size; idx += blockDim.x) {
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

    int threads = next_power_of_two(dim_size);
    threads = (threads + 31) / 32 * 32;
    threads = threads < 1024 ? threads : 1024;

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