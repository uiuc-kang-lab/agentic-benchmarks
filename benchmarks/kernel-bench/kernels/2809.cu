#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 256;
const int ELEMENTS_PER_THREAD = 4;
const int SHARED_MEM_SIZE = THREADS * ELEMENTS_PER_THREAD;

template <typename scalar_t>
__forceinline__ __device__ float sigmoid_compute(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    __shared__ float shared_data[SHARED_MEM_SIZE];
    
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * SHARED_MEM_SIZE;
    
    // Vectorized loading into shared memory
    using float4_t = float4;
    if (tid * ELEMENTS_PER_THREAD + ELEMENTS_PER_THREAD <= size) {
        float4_t* in_ptr = (float4_t*)&input[block_offset + tid * ELEMENTS_PER_THREAD];
        float4_t* shared_ptr = (float4_t*)&shared_data[tid * ELEMENTS_PER_THREAD];
        *shared_ptr = *in_ptr;
    } else {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int idx = block_offset + tid + i * THREADS;
            if (idx < size) {
                shared_data[tid + i * THREADS] = static_cast<float>(input[idx]);
            }
        }
    }
    __syncthreads();
    
    // Process elements using optimized sigmoid computation
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = block_offset + tid + i * THREADS;
        if (idx < size) {
            float val = shared_data[tid + i * THREADS];
            output[idx] = static_cast<scalar_t>(sigmoid_compute<scalar_t>(val));
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int blocks = (size + SHARED_MEM_SIZE - 1) / SHARED_MEM_SIZE;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t><<<blocks, THREADS>>>(input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA)");
}