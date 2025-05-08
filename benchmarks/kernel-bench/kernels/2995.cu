#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_shared_vector_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    constexpr int VEC_SIZE = 4;
    __shared__ scalar_t tile[128]; // 32 threads * 4 elements
    
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * 128;
    
    // Start asynchronous load into shared memory using cooperative groups
    auto block = cooperative_groups::this_thread_block();
    float4 vec_in;
    if (block_offset + tid * VEC_SIZE < size) {
        vec_in = reinterpret_cast<const float4*>(input + block_offset)[tid];
    }
    
    // Start processing previous elements while waiting for shared memory
    scalar_t results[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        results[i] = tanhf(((const float*)&vec_in)[i]);
    }
    
    // Store to shared memory for potential reuse by other threads
    if (block_offset + tid * VEC_SIZE < size) {
        reinterpret_cast<float4*>(tile)[tid] = vec_in;
    }
    block.sync();
    
    // Vectorized store from registers
    if (block_offset + tid * VEC_SIZE < size) {
        float4 vec_out = {results[0], results[1], results[2], results[3]};
        reinterpret_cast<float4*>(output + block_offset)[tid] = vec_out;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 32; // Processes 128 elements per block
    const int blocks = (input.numel() + 127) / 128;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_shared_vector_kernel", ([&] {
        tanh_shared_vector_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with vectorized shared memory (CUDA)");
}