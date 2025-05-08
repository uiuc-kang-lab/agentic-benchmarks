#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

template<typename scalar_t>
__device__ __forceinline__ scalar_t hardsigmoid_op(scalar_t x) {
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = 1.0/6.0;
    x = fma(x, sixth, three * sixth);
    return fmaxf(0.0f, fminf(1.0f, x));
}

template<typename scalar_t, int VEC_SIZE>
__device__ __forceinline__ void load_vector(scalar_t* dst, const scalar_t* src) {
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value, float4,
        typename std::conditional<std::is_same<scalar_t, double>::value, double2, void>::type
    >::type;
    *reinterpret_cast<vec_t*>(dst) = *reinterpret_cast<const vec_t*>(src);
}

template<typename scalar_t, int VEC_SIZE>
__device__ __forceinline__ void store_vector(scalar_t* dst, const scalar_t* src) {
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value, float4,
        typename std::conditional<std::is_same<scalar_t, double>::value, double2, void>::type
    >::type;
    *reinterpret_cast<vec_t*>(dst) = *reinterpret_cast<const vec_t*>(src);
}

template<typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    const size_t num_vectors = numel / VEC_SIZE;
    const size_t vector_offset = blockDim.x * gridDim.x;
    
    // Process vectorized elements
    for(size_t vid = blockIdx.x * blockDim.x + threadIdx.x; 
        vid < num_vectors; 
        vid += vector_offset) {
        scalar_t elements[VEC_SIZE];
        load_vector<scalar_t, VEC_SIZE>(elements, &input[vid * VEC_SIZE]);
        
        #pragma unroll
        for(int i = 0; i < VEC_SIZE; ++i) {
            elements[i] = hardsigmoid_op(elements[i]);
        }
        
        store_vector<scalar_t, VEC_SIZE>(&output[vid * VEC_SIZE], elements);
    }

    // Process remaining elements
    const size_t tail_start = num_vectors * VEC_SIZE;
    for(size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; 
        i < numel; 
        i += vector_offset) {
        output[i] = hardsigmoid_op(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();

    constexpr int threads = 256;
    const int max_blocks = (numel + threads - 1) / threads;
    const int blocks = min(max_blocks, 128);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", [&] {
        constexpr int vec_size = std::is_same<scalar_t, float>::value ? 4 : 2;
        hardsigmoid_kernel<scalar_t, vec_size><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular HardSigmoid forward (CUDA)");
}
