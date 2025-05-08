#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Calculate input position with padding
__device__ __forceinline__ int get_input_pos(int output_pos, int k, int stride, int padding) {
    return output_pos * stride + k - padding;
}

// Calculate global memory index for input tensor
__device__ __forceinline__ int get_input_index(
    int batch, int channel, int pos, 
    int in_channels, int input_length) {
    return batch * (in_channels * input_length) + channel * input_length + pos;
}

// Calculate global memory index for output tensor
__device__ __forceinline__ int get_output_index(
    int batch, int channel, int pos, 
    int in_channels, int output_length) {
    return batch * (in_channels * output_length) + channel * output_length + pos;
}

// Compute average over kernel window
__device__ __forceinline__ float compute_window_average(
    const float* input,
    int start_pos,
    int input_length,
    int kernel_size,
    int batch,
    int channel,
    int in_channels) {
    
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int pos_input = start_pos + k;
        if (pos_input >= 0 && pos_input < input_length) {
            sum += input[get_input_index(batch, channel, pos_input, in_channels, input_length)];
        }
    }
    return sum / kernel_size;
}

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

    const int o = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;

    if (o >= output_length || channel >= in_channels || batch >= batch_size) 
        return;

    // Calculate starting position in input tensor considering padding
    int start_pos = get_input_pos(o, 0, stride, padding);
    
    // Compute average for current window
    float result = compute_window_average(
        input, start_pos, input_length, kernel_size,
        batch, channel, in_channels
    );

    // Write result to output tensor
    output[get_output_index(batch, channel, o, in_channels, output_length)] = result;
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

    // Maintain the effective 3D thread configuration
    dim3 threads(32, 8, 4);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        (in_channels + threads.y - 1) / threads.y,
        (batch_size + threads.z - 1) / threads.z
    );

    avg_pool1d_kernel<<<grid, threads>>>(
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
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA)");
}