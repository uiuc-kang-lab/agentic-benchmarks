#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Constant memory declarations
__constant__ int c_dim_size;
__constant__ int c_thread_count;

inline int next_power_of_two(int x) {
    return 1 << (32 - __builtin_clz(x - 1));
}

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const scalar_t* __restrict__ input_row = input + batch_idx * c_dim_size;
    scalar_t* output_row = output + batch_idx * c_dim_size;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Find max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    #pragma unroll 4
    for (int idx = tid; idx < c_dim_size; idx += c_thread_count) {
        max_val = max(max_val, __ldg(input_row + idx));
    }
    
    sdata[tid] = max_val; __syncthreads();
    __syncthreads();

    for (unsigned int stride = c_thread_count / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    max_val = sdata[0];

    // Compute exp sum without intermediate storage
    scalar_t sum = 0;
    #pragma unroll 4
    for (int idx = tid; idx < c_dim_size; idx += c_thread_count) {
        sum += exp(__ldg(input_row + idx) - max_val);
    }
    
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = c_thread_count / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    scalar_t log_sum = log(sdata[0]);

    #pragma unroll 4
    for (int idx = tid; idx < c_dim_size; idx += c_thread_count) {
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

    // Copy constants to constant memory
    int h_dim_size = static_cast<int>(dim_size);
    int h_thread_count = threads;
    cudaMemcpyToSymbol(c_dim_size, &h_dim_size, sizeof(int));
    cudaMemcpyToSymbol(c_thread_count, &h_thread_count, sizeof(int));

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        size_t shared_mem_size = threads * sizeof(scalar_t);
        log_softmax_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
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