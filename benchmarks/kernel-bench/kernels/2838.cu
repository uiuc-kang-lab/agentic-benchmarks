#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device function for exponentiation preserving precision
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

// Union for vectorized load/store using 128-bit accesses
template <typename scalar_t, typename VecT, int VecSize>
union VecUnion {
    VecT vec;
    scalar_t arr[VecSize];
};

// Kernel for vectorized sigmoid computation using grid-stride loop
template <typename scalar_t, typename VecT, int VecSize>
__global__ void vectorized_sigmoid_kernel(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             int64_t vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < vec_count; i += stride) {
        VecUnion<scalar_t, VecT, VecSize> in_val;
        VecUnion<scalar_t, VecT, VecSize> out_val;
        // Use __ldg() for read-only global memory access, assuming proper alignment
        in_val.vec = __ldg(reinterpret_cast<const VecT*>(input) + i);

        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            scalar_t val = in_val.arr[j];
            scalar_t exp_val = myExp(-val);
            out_val.arr[j] = (scalar_t(1)) / (scalar_t(1) + exp_val);
        }
        reinterpret_cast<VecT*>(output)[i] = out_val.vec;
    }
}

// Scalar kernel to handle the tail elements
template <typename scalar_t>
__global__ void scalar_sigmoid_kernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        int64_t start,
                                        int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < size) {
        scalar_t val = __ldg(&input[idx]);
        scalar_t exp_val = myExp(-val);
        output[idx] = (scalar_t(1)) / (scalar_t(1) + exp_val);
    }
}

// Forward function launching the optimized kernels
// This function vectorizes accesses when possible using 128-bit loads/stores
// and falls back to a scalar kernel for any remaining tail elements.

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 512;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vectorized_sigmoid_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        // Choose vectorization factor based on scalar type: float4 for float, double2 for double
        int vecSize = std::is_same<scalar_t, float>::value ? 4 : 2;
        int64_t vec_elements = size / vecSize; // Number of full vectorized groups
        int blocks = (vec_elements + threads - 1) / threads;

        if (vec_elements > 0) {
            if (std::is_same<scalar_t, float>::value) {
                vectorized_sigmoid_kernel<scalar_t, float4, 4><<<blocks, threads, 0, stream>>>(input_data, output_data, vec_elements);
            } else {
                vectorized_sigmoid_kernel<scalar_t, double2, 2><<<blocks, threads, 0, stream>>>(input_data, output_data, vec_elements);
            }
        }

        // Process tail elements that don't fit into a full 128-bit vector
        int64_t vec_aligned_size = vec_elements * vecSize;
        int64_t tail = size - vec_aligned_size;
        if (tail > 0) {
            int tail_blocks = (tail + threads - 1) / threads;
            scalar_sigmoid_kernel<scalar_t><<<tail_blocks, threads, 0, stream>>>(input_data, output_data, vec_aligned_size, size);
        }
    }));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Streamed vectorized sigmoid forward (CUDA)");
}
