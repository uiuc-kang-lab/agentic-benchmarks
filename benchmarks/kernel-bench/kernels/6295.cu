#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D average pooling (count_include_pad=True) using 3D grid and block indexing
// This kernel maps the 5D output tensor [N, C, D_out, H_out, W_out] onto a 3D grid by fusing the D_out, batch, and channel dimensions.
// Each thread computes one output element, reducing index arithmetic overhead compared to a 1D-stride loop.

__global__ void avg_pool3d_forward_kernel_3D(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Calculate thread coordinates for the output spatial dimensions (W and H) and fused D, N, and C dimension
    int w_out = threadIdx.x + blockIdx.x * blockDim.x;  // output width index
    int h_out = threadIdx.y + blockIdx.y * blockDim.y;  // output height index
    int d_nc = threadIdx.z + blockIdx.z * blockDim.z;     // fused dimension for d_out, n, and c

    // Total fused length for (batch, channel, d_out)
    int fused_size = out_d * batch_size * channels;

    // Check bounds
    if (w_out >= out_w || h_out >= out_h || d_nc >= fused_size) return;

    // Decode the fused dimension: d_out, batch index (n) and channel (c)
    int d_out = d_nc % out_d;
    int bc = d_nc / out_d;
    int n = bc / channels;
    int c = bc % channels;

    // Calculate pooling window boundaries (including padding)
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    int d_end = d_start + kernel_size;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    // Clamp window boundaries to input dimensions
    int d_start_clamped = (d_start < 0) ? 0 : d_start;
    int h_start_clamped = (h_start < 0) ? 0 : h_start;
    int w_start_clamped = (w_start < 0) ? 0 : w_start;
    int d_end_clamped = (d_end > in_d) ? in_d : d_end;
    int h_end_clamped = (h_end > in_h) ? in_h : h_end;
    int w_end_clamped = (w_end > in_w) ? in_w : w_end;

    float sum = 0.0f;
    // Sum over the pooling window
    for (int d = d_start_clamped; d < d_end_clamped; ++d) {
        for (int h = h_start_clamped; h < h_end_clamped; ++h) {
            for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                int input_index = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
                sum += input[input_index];
            }
        }
    }

    // For count_include_pad=True, division is by full kernel volume
    int pool_volume = kernel_size * kernel_size * kernel_size;
    float avg = __fdividef(sum, static_cast<float>(pool_volume));

    // Compute the linear index for the output tensor in [N, C, D_out, H_out, W_out]
    int output_index = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
    output[output_index] = avg;
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    // Check input tensor properties
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Compute output dimensions based on pooling formula
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Configure 3D block and grid dimensions
    // Block dimensions: x covers width, y covers height, z covers fused (d_out * batch_size * channels)
    dim3 block(16, 16, 4);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              ((out_d * batch_size * channels) + block.z - 1) / block.z);

    avg_pool3d_forward_kernel_3D<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with 3D grid mapping");
}
