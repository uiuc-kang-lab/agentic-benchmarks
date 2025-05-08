#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_tuned(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better work distribution
    for (int i = idx; i < size; i += stride) {
        scalar_t val = input[i];
        output[i] = val > 0 ? val : 0;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Optimized block size of 128 threads
    const int threads = 128;
    // Ensure enough blocks to fully utilize GPU
    const int min_blocks_per_sm = 2;
    const int device_id = input.device().index();
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
    const int blocks = min(
        (input.numel() + threads - 1) / threads,
        num_sms * min_blocks_per_sm
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_tuned", ([&] {
        relu_kernel_tuned<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with tuned block size (CUDA)");
}