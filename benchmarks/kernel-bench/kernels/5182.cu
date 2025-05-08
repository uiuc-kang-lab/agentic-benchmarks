#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <type_traits>

#define MAX_NORMALIZED_SIZE 4096

// Declare constant memory for read-only weight and bias data
__constant__ float const_weight_float[MAX_NORMALIZED_SIZE];
__constant__ float const_bias_float[MAX_NORMALIZED_SIZE];

__constant__ double const_weight_double[MAX_NORMALIZED_SIZE];
__constant__ double const_bias_double[MAX_NORMALIZED_SIZE];

// Kernel using 2D thread blocks and constant memory for weight and bias
template <typename scalar_t>
__global__ void layernorm_2d_const_kernel(
    const scalar_t* __restrict__ input,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size
) {
    using accscalar_t = at::acc_type<scalar_t, true>;

    // 2D thread block indices
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int instance_idx = blockIdx.x;  // each block handles one instance

    // Total threads per block
    const int num_threads = blockDim.x * blockDim.y;
    const int thread_id = tidy * blockDim.x + tidx;

    // Allocate shared memory for partial reductions (sum and sum_sq)
    extern __shared__ extern __shared__ accscalar_t sdata[];
    accscalar_t* s_sum = sdata;             // first half for sum
    accscalar_t* s_sum_sq = sdata + num_threads; // second half for sum of squares

    // Pointers to this instance's data
    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;

    // Each thread processes elements in a strided manner
    for (int idx = thread_id; idx < normalized_size; idx += num_threads) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[thread_id] = local_sum;
    s_sum_sq[thread_id] = local_sum_sq;
    __syncthreads();

    // Reduce partial sums using iterative halving
    for (int stride = num_threads / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            s_sum[thread_id] += s_sum[thread_id + stride];
            s_sum_sq[thread_id] += s_sum_sq[thread_id + stride];
        }
        __syncthreads();
    }

    // Compute mean and inverse standard deviation
    __shared__ accscalar_t mean, inv_std;
    if (thread_id == 0) {
        mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
        accscalar_t variance = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
        inv_std = rsqrt(variance + static_cast<accscalar_t>(eps));
    }
    __syncthreads();

    // Normalize and apply affine transformation using constant memory for weight and bias
    for (int idx = thread_id; idx < normalized_size; idx += num_threads) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        accscalar_t norm = (val - mean) * inv_std;
        if constexpr (std::is_same<scalar_t, float>::value) {
            out_ptr[idx] = static_cast<scalar_t>(norm * const_weight_float[idx] + const_bias_float[idx]);
        } else {
            out_ptr[idx] = static_cast<scalar_t>(norm * const_weight_double[idx] + const_bias_double[idx]);
        }
    }
}

// Host function to copy weight and bias to constant memory, then launch the kernel
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;

    // Ensure the normalized dimension fits within our constant memory allocation
    TORCH_CHECK(normalized_size <= MAX_NORMALIZED_SIZE, "normalized_size exceeds maximum constant memory capacity");

    // Copy weight and bias to constant memory
    if (x.scalar_type() == at::kFloat) {
        cudaMemcpyToSymbol(const_weight_float, weight.data_ptr<float>(), normalized_size * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(const_bias_float, bias.data_ptr<float>(), normalized_size * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    } else { // assume double
        cudaMemcpyToSymbol(const_weight_double, weight.data_ptr<double>(), normalized_size * sizeof(double), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(const_bias_double, bias.data_ptr<double>(), normalized_size * sizeof(double), 0, cudaMemcpyDeviceToDevice);
    }

    // Configure kernel launch parameters
    // Using a 2D thread block (e.g., 16x16 threads per block for good occupancy)
    dim3 threads(16, 16);
    dim3 blocks(outer_size);
    int num_threads = threads.x * threads.y;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_const_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        int shared_mem_size = num_threads * 2 * sizeof(accscalar_t);
        layernorm_2d_const_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            static_cast<float>(eps),
            output.data_ptr<scalar_t>(),
            normalized_size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward with constant memory (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
