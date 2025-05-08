#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int KERNEL_SIZE>
__global__ void optimized_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {
    
    extern __shared__ float shared_input[];
    
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    const int batch = blockIdx.z;
    
    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;
    
    const int start_idx = o * stride - padding;
    const int end_idx = start_idx + KERNEL_SIZE;
    const float scale = 1.0f / KERNEL_SIZE;
    
    float sum = 0.0f;
    
    const int base_idx = batch * in_channels * input_length + channel * input_length;
    
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        const int pos_input = start_idx + k;
        if (pos_input >= 0 && pos_input < input_length) {
            sum += input[base_idx + pos_input];
        }
    }
    
    output[batch * in_channels * output_length + channel * output_length + o] = sum * scale;
}

torch::Tensor optimized_avg_pool1d_forward(
    const torch::Tensor &x,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid parameters");
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());
    
    const int threads = 256;
    dim3 blocks(
        (output_length + threads - 1) / threads,
        in_channels,
        batch_size
    );
    
    switch(kernel_size) {
        case 2:
            optimized_avg_pool1d_kernel<2><<<blocks, threads>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                stride, padding, input_length, output_length,
                batch_size, in_channels);
            break;
        case 3:
            optimized_avg_pool1d_kernel<3><<<blocks, threads>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                stride, padding, input_length, output_length,
                batch_size, in_channels);
            break;
        case 4:
            optimized_avg_pool1d_kernel<4><<<blocks, threads>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                stride, padding, input_length, output_length,
                batch_size, in_channels);
            break;
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_avg_pool1d_forward, "Optimized 1D Average Pooling forward (CUDA)");
}