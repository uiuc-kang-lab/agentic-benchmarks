#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

template <typename scalar_t>
struct vectorized_traits {
    using vec_t = void;
    static const int width = 1;
};

template <>
struct vectorized_traits<float> {
    using vec_t = float4;
    static const int width = 4;
    __device__ static float4 load(const float* ptr) {
        return __ldg(reinterpret_cast<const float4*>(ptr));
    }
    __device__ static void store(float* ptr, float4 val) {
        *reinterpret_cast<float4*>(ptr) = val;
    }
    __device__ static float4 tanh_vec(float4 in) {
        return make_float4(tanhf(in.x), tanhf(in.y), tanhf(in.z), tanhf(in.w));
    }
};

template <>
struct vectorized_traits<double> {
    using vec_t = double2;
    static const int width = 2;
    __device__ static double2 load(const double* ptr) {
        return *reinterpret_cast<const double2*>(ptr);
    }
    __device__ static void store(double* ptr, double2 val) {
        *reinterpret_cast<double2*>(ptr) = val;
    }
    __device__ static double2 tanh_vec(double2 in) {
        return make_double2(tanh(in.x), tanh(in.y));
    }
};

template <typename scalar_t>
__global__ void optimized_tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    using traits = vectorized_traits<scalar_t>;
    const int vec_width = traits::width;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Handle vectorized elements
    #pragma unroll
    for (int i = tid; i < size / vec_width; i += stride) {
        auto vec_in = traits::load(input + i * vec_width);
        auto vec_out = traits::tanh_vec(vec_in);
        traits::store(output + i * vec_width, vec_out);
    }
    
    // Handle remaining elements using collaborative warp processing
    const int remainder_start = (size / vec_width) * vec_width;
    const int lane_id = threadIdx.x % warpSize;
    
    for (int i = remainder_start + lane_id; i < size; i += warpSize) {
        if (i < size) {
            output[i] = std::is_same<scalar_t, float>::value ? tanhf(input[i]) : tanh(input[i]);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize thread and block count
    const int threads = 256;
    const int max_blocks = 1024;
    const int min_blocks_needed = (size + threads - 1) / threads;
    const int blocks = std::min(max_blocks, min_blocks_needed);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_tanh_kernel", ([&] {
        optimized_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized vectorized Tanh forward (CUDA)");
}