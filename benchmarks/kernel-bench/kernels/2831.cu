#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Define an inline device function for exponentiation, specialized for float and double.

template <typename T>
__device__ inline T myExp(T x);

template <>
__device__ inline float myExp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double myExp<double>(double x) {
    return exp(x);
}

// Union to facilitate vectorized load and store operations
// VecT: vector type (e.g., float4 or double2), VecSize: number of scalar elements in VecT

template <typename scalar_t, typename VecT, int VecSize>
union VecUnion {
  VecT vec;
  scalar_t arr[VecSize];
};

// Vectorized kernel processing multiple elements per thread using 128-bit loads/stores
// It uses __ldg() to optimize read-only global memory accesses.

template <typename scalar_t, typename VecT, int VecSize>
__global__ void sigmoid_vectorized_kernel(const scalar_t* __restrict__ input,
                                            scalar_t* __restrict__ output,
                                            int64_t vec_count) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vec_count; idx += blockDim.x * gridDim.x) {
        VecUnion<scalar_t, VecT, VecSize> in_union;
        VecUnion<scalar_t, VecT, VecSize> out_union;
        
        // Load a 128-bit chunk from global memory (assumed to be aligned), using __ldg for read-only access
        in_union.vec = __ldg(reinterpret_cast<const VecT*>(input) + idx);
        
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            scalar_t val = in_union.arr[i];
            scalar_t exp_val = myExp(-val);
            out_union.arr[i] = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + exp_val);
        }

        // Store the computed vector back to global memory (assumed aligned)
        reinterpret_cast<VecT*>(output)[idx] = out_union.vec;
    }
}

// Scalar kernel for processing tail elements that don't fit in a full vectorized load/store

template <typename scalar_t>
__global__ void sigmoid_scalar_kernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        int64_t start,
                                        int64_t size) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
         idx < size;
         idx += blockDim.x * gridDim.x) {
        scalar_t val = __ldg(&input[idx]);
        scalar_t exp_val = myExp(-val);
        output[idx] = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + exp_val);
    }
}

// The forward function prepares the output tensor and launches the appropriate kernels
// It handles vectorized processing for 128-bit aligned data and falls back to a scalar kernel for tail elements.

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_vectorized_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();

        // Determine the vectorization factor and vector type based on the scalar type
        int vecSize = 1;
        int64_t vec_elements = 0;
        int blocks = 0;

        if (std::is_same<scalar_t, float>::value) {
            vecSize = 4; // 128-bit: 4 x float
            vec_elements = size / vecSize; // number of full vectorized groups
            blocks = (vec_elements + threads - 1) / threads;
            if (vec_elements > 0) {
                sigmoid_vectorized_kernel<scalar_t, float4, 4><<<blocks, threads>>>(input_data, output_data, vec_elements);
            }
        } else if (std::is_same<scalar_t, double>::value) {
            vecSize = 2; // 128-bit: 2 x double
            vec_elements = size / vecSize;
            blocks = (vec_elements + threads - 1) / threads;
            if (vec_elements > 0) {
                sigmoid_vectorized_kernel<scalar_t, double2, 2><<<blocks, threads>>>(input_data, output_data, vec_elements);
            }
        }
        
        // Process any remaining tail elements not covered by vectorized loads/stores
        int64_t vec_aligned_size = vec_elements * vecSize;
        int64_t tail = size - vec_aligned_size;
        if (tail > 0) {
            int tail_blocks = (tail + threads - 1) / threads;
            sigmoid_scalar_kernel<scalar_t><<<tail_blocks, threads>>>(input_data, output_data, vec_aligned_size, size);
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized Sigmoid forward (CUDA) with vectorized load/store");
}
