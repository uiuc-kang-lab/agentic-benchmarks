#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    extern __shared__ float smem[];
    
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    const int batch = blockIdx.z;

    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    // Calculate input window for this block
    const int block_start = blockIdx.x * blockDim.x * stride - padding;
    const int block_end = block_start + (blockDim.x * stride) + kernel_size - 1;
    const int smem_size = block_end - block_start;

    // Cooperative loading into shared memory
    for (int i = threadIdx.x; i < smem_size; i += blockDim.x) {
        int global_pos = block_start + i;
        smem[i] = (global_pos >= 0 && global_pos < input_length) ? 
            input[batch * in_channels * input_length + channel * input_length + global_pos] : 0.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    const int window_start = o * stride - padding - block_start;
    
    #pragma unroll 4
    for (int k = 0; k < kernel_size; ++k) {
        if (window_start + k >= 0 && window_start + k < smem_size) {
            sum += smem[window_start + k];
        }
    }

    output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
}

torch::Tensor avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    const int block_size = 256;
    dim3 blocks(
        (output_length + block_size - 1) / block_size,
        in_channels,
        batch_size
    );

    // Calculate shared memory size for worst-case block input window
    const int smem_size = (block_size * stride + kernel_size - 1) * sizeof(float);

    avg_pool1d_kernel<<<blocks, block_size, smem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with shared memory coalescing");
}
