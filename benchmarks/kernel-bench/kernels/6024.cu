#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__global__ void avg_pool1d_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
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

    // Calculate base indices for input and output
    const int input_batch_offset = batch * in_channels * input_length;
    const int input_channel_offset = channel * input_length;
    const int base_input_idx = input_batch_offset + input_channel_offset;

    float sum = 0.0f;
    
    // Align the starting position to a 16-byte boundary where possible
    const int window_start = o * stride - padding;
    const int aligned_start = (window_start + 3) & ~3;
    const int aligned_end = (window_start + kernel_size) & ~3;

    // Handle pre-aligned elements
    #pragma unroll
    for (int k = window_start; k < aligned_start && k < window_start + kernel_size; ++k) {
        if (k >= 0 && k < input_length) {
            sum += __ldg(&input[base_input_idx + k]);
        }
    }

    // Process aligned elements using float4
    for (int k = aligned_start; k < aligned_end; k += 4) {
        if (k >= 0 && k + 3 < input_length && k < window_start + kernel_size) {
            float4 values = load_float4(&input[base_input_idx + k]);
            sum += values.x + values.y + values.z + values.w;
        }
    }

    // Handle post-aligned elements
    #pragma unroll
    for (int k = aligned_end; k < window_start + kernel_size; ++k) {
        if (k >= 0 && k < input_length) {
            sum += __ldg(&input[base_input_idx + k]);
        }
    }

    // Write result using aligned store when possible
    const int output_idx = batch * in_channels * output_length + 
                          channel * output_length + o;
    output[output_idx] = sum / kernel_size;
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

    // Optimize thread configuration for aligned access
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