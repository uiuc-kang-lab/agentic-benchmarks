#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel_tiled(
    const float4 *input4,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    float sum = 0.0f;
    int input_start = o * stride - padding;
    
    // Calculate aligned positions for vectorized loading
    int aligned_start = (input_start + 3) / 4 * 4;
    int aligned_end = (input_start + kernel_size) / 4 * 4;
    
    // Handle pre-aligned elements
    for (int pos = input_start; pos < aligned_start && pos < input_start + kernel_size; pos++) {
        if (pos >= 0 && pos < input_length) {
            sum += reinterpret_cast<const float*>(input4)[batch * in_channels * input_length + channel * input_length + pos];
        }
    }

    // Vectorized loading of aligned elements
    for (int pos = aligned_start; pos < aligned_end; pos += 4) {
        if (pos >= 0 && pos + 3 < input_length && pos < input_start + kernel_size) {
            float4 data = input4[(batch * in_channels * input_length + channel * input_length + pos) / 4];
            sum += data.x + data.y + data.z + data.w;
        }
    }

    // Handle post-aligned elements
    for (int pos = aligned_end; pos < input_start + kernel_size; pos++) {
        if (pos >= 0 && pos < input_length) {
            sum += reinterpret_cast<const float*>(input4)[batch * in_channels * input_length + channel * input_length + pos];
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

    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    avg_pool1d_kernel_vectorized<<<grid, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
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
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with vectorized loading (CUDA)");
}