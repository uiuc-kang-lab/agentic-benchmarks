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

// Customized kernel that allows flexible block sizes
// optimized for Sigmoid function
template <typename scalar_t, typename VecT, int VecSize>
__global__ void customized_sigmoid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = sizeof(VecT) / sizeof(scalar_t);
    const int64_t vec_elements = size / vec_size;

    // Vectorized processing
    for (int64_t i = tid; i < vec_elements; i += stride) {
        VecUnion<scalar_t, VecT, VecSize> in_union;
        VecUnion<scalar_t, VecT, VecSize> out_union;
        
        // Use __ldg for read-only cache optimized load
        in_union.vec = __ldg(reinterpret_cast<const VecT*>(input) + i);
        
        // Process vector elements using fast math
        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            scalar_t val = -in_union.arr[j];
            scalar_t exp_val = myExp(val);
            out_union.arr[j] = scalar_t(1) / (scalar_t(1) + exp_val);
        }
        
        // Aligned vector store
        reinterpret_cast<VecT*>(output)[i] = out_union.vec;
    }

    // Handle remaining elements
    const int64_t vec_offset = vec_elements * vec_size;
    for (int64_t i = vec_offset + tid; i < size; i += stride) {
        scalar_t val = __ldg(&input[i]);
        scalar_t exp_val = myExp(-val);
        output[i] = scalar_t(1) / (scalar_t(1) + exp_val);
    }
}

// Forward function launching the optimized kernels
// This function vectorizes accesses when possible using 128-bit loads/stores
// and falls back to a scalar kernel for any remaining tail elements.

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int candidate_threads[] = {32, 64, 128, 256, 512};  // Explore these block sizes
    const int optimal_threads = 512;  // Initial assumption; further tuning might be necessary
    const int threads = optimal_threads;
    const int max_blocks = 65535;
    const int min_elements_per_thread = 4;
    const int blocks = static_cast<int>(std::min(static_cast<int64_t>(max_blocks),
                         (size + threads * min_elements_per_thread - 1) / 
                         (threads * min_elements_per_thread)));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "customized_sigmoid_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();

        if (std::is_same<scalar_t, float>::value) {
            customized_sigmoid_kernel<scalar_t, float4, 4>
                <<<blocks, threads>>>(input_data, output_data, size);
        } else {
            customized_sigmoid_kernel<scalar_t, double2, 2>
                <<<blocks, threads>>>(input_data, output_data, size);
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Customizable Sigmoid forward (CUDA)");
}
