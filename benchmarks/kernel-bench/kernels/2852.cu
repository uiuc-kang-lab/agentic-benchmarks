#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device function for exponentiation preserving precision
template <typename T>
__device__ inline T myExp(T x);

template <>
__device__ inline float myExp<float>(float x) {
    return __expf(x);  // Using faster CUDA math intrinsic
}

template <>
__device__ inline double myExp<double>(double x) {
    return exp(x);
}

// Union for vectorized load/store using 128-bit accesses
template <typename scalar_t, typename VecT, int VecSize>
union VecUnion {
    VecT vec;
    scalar_t arr[VecSize];
};

// Optimized kernel for computing sigmoid using a grid-stride loop
template <typename scalar_t, typename VecT, int VecSize>
__global__ void balanced_sigmoid_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          const int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process multiple elements per thread to ensure balanced workload
    for (int64_t i = idx; i < size; i += stride) {
        scalar_t x = __ldg(&input[i]);
        scalar_t exp_val = myExp(-x);
        output[i] = scalar_t(1) / (scalar_t(1) + exp_val);
    }
}

// The forward function that sets up the kernel launch configuration and takes care of processing
// It uses a grid-stride loop in the kernel to distribute work evenly across blocks and threads

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 512;  // Increased threads for more parallelism
    const int max_blocks_per_sm = 32;  // Hypothetical maximum for better GPU utilization

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "balanced_sigmoid_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();

        int blocks = std::min(max_blocks_per_sm, (size + threads - 1) / threads);

        if (std::is_same<scalar_t, float>::value) {
            balanced_sigmoid_kernel<scalar_t, float, 1><<<blocks, threads>>>(input_data, output_data, size);
        } else {
            balanced_sigmoid_kernel<scalar_t, double, 1><<<blocks, threads>>>(input_data, output_data, size);
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Sigmoid forward (CUDA)");
}