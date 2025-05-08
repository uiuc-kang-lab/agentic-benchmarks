#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Custom atomicMax for float
__device__ inline float atomicMaxCustom(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(val, old_val);
        int new_val_int = __float_as_int(new_val);
        old = atomicCAS(address_as_i, assumed, new_val_int);
    } while (assumed != old);
    return __int_as_float(old);
}

// Custom atomicMax for double
__device__ inline double atomicMaxCustom(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmax(val, old_val);
        unsigned long long new_val_ull = __double_as_longlong(new_val);
        old = atomicCAS(address_as_ull, assumed, new_val_ull);
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel using atomics to perform block-level reduction without divergent branching
template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Allocate two scalar_t values in shared memory: one for block max and one for block sum
    extern __shared__ char smem[];
    scalar_t* smem_f = reinterpret_cast<scalar_t*>(smem);
    // smem_f[0] holds the block max; smem_f[1] holds the block sum
    if (tid == 0) {
        smem_f[0] = -std::numeric_limits<scalar_t>::infinity();
        smem_f[1] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    // Each thread computes a local maximum over its portion
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = tid; idx < dim_size; idx += blockDim.x) {
        local_max = max(local_max, input_row[idx]);
    }
    // Update the block maximum using an atomic operation (all threads call this uniformly)
    atomicMaxCustom(&smem_f[0], local_max);
    __syncthreads();
    scalar_t global_max = smem_f[0];

    // Each thread computes the partial sum of exp(input - global_max)
    scalar_t local_sum = 0;
    for (int idx = tid; idx < dim_size; idx += blockDim.x) {
        local_sum += exp(input_row[idx] - global_max);
    }
    // Accumulate the block sum using atomicAdd
    atomicAdd(&smem_f[1], local_sum);
    __syncthreads();
    scalar_t global_sum = smem_f[1];
    scalar_t log_sum = log(global_sum);

    // Write final output
    for (int idx = tid; idx < dim_size; idx += blockDim.x) {
        output_row[idx] = (input_row[idx] - global_max) - log_sum;
    }
}

// Host function
torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute so that target dim is the last dimension
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

    // Choose the number of threads: next power of two up to 1024
    int threads = 1;
    while (threads < dim_size && threads < 1024) {
        threads *= 2;
    }

    // Shared memory: 2 scalars (block max and block sum)
    size_t shared_mem_size = 2 * ((input.scalar_type() == torch::kFloat64) ? sizeof(double) : sizeof(float));

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    // Inverse-permute to restore the original tensor order
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
