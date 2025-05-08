#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void tanh_kernel_pipelined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int chunk_size,
    const int offset) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        const int global_idx = offset + idx;
        float4 in = reinterpret_cast<const float4*>(input)[global_idx/4];
        float4 out;
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
        reinterpret_cast<float4*>(output)[global_idx/4] = out;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int num_elements = input.numel();
    const int num_streams = 4;
    int chunk_size = (num_elements + num_streams - 1) / num_streams;
    chunk_size = (chunk_size + 3) & ~3; // Ensure chunk_size is multiple of 4
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int threads = 256;
    const int blocks_per_chunk = (chunk_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_kernel_pipelined", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const int offset = i * chunk_size;
            const int current_chunk_size = std::min(chunk_size, num_elements - offset);
            if (current_chunk_size > 0) {
                tanh_kernel_pipelined<scalar_t><<<blocks_per_chunk, threads, 0, streams[i]>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    current_chunk_size,
                    offset
                );
            }
        }
    }));
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined Tanh forward (CUDA)");
}