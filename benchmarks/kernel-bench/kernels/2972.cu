#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Define vectorized traits for float and double to enable vectorized load/stores

template <typename scalar_t>
struct vectorized_traits;

// Specialization for float using float4 (4 floats, 16 bytes) -> enhances memory coalescing
template <>
struct vectorized_traits<float> {
    using vec_t = float4;
    static const int width = 4;
    __device__ static void apply(const vec_t &in, vec_t &out) {
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
    }
};

// Specialization for double using double2 (2 doubles, 16 bytes)
template <>
struct vectorized_traits<double> {
    using vec_t = double2;
    static const int width = 2;
    __device__ static void apply(const vec_t &in, vec_t &out) {
        out.x = tanh(in.x);
        out.y = tanh(in.y);
    }
};

// Device-specific tanh function: use tanhf for float and tanh for double.
template <typename scalar_t>
__device__ inline scalar_t device_tanh(scalar_t x);

template <>
__device__ inline float device_tanh<float>(float x) {
    return tanhf(x);
}

template <>
__device__ inline double device_tanh<double>(double x) {
    return tanh(x);
}

// Kernel using stride loop for better warp efficiency and memory coalescing
// The kernel uses vectorized operations for the main workload and falls back to scalar operations for the remainder
// It utilizes stride loops to efficiently handle larger workloads

template <typename scalar_t>
__global__ void tanh_kernel_stride_loop(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int size) {

    using traits = vectorized_traits<scalar_t>;
    using vec_t = typename traits::vec_t;
    constexpr int vec_width = traits::width;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process vectorized portion: each load/store handles vec_width elements
    for (int i = idx; i < size / vec_width; i += stride) {
        vec_t vec_in = reinterpret_cast<const vec_t*>(input)[i];
        vec_t vec_out;
        traits::apply(vec_in, vec_out);
        reinterpret_cast<vec_t*>(output)[i] = vec_out;
    }

    // Process remaining elements that do not fit in a vectorized load/store
    int rem_start = (size / vec_width) * vec_width;
    for (int i = rem_start + idx; i < size; i += stride) {
        output[i] = device_tanh(input[i]);
    }
}

// Host function to launch the stride-loop optimized CUDA kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_stride_loop", ([&] {
        tanh_kernel_stride_loop<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride-loop optimized Tanh forward (CUDA)");
}