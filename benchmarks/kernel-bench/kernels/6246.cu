#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Use blockIdx.z for (n, c, d_out) combined
    int idx = blockIdx.z;
    int d_out = idx % out_d;
    idx /= out_d;
    int c = idx % channels;
    int n = idx / channels;

    // Compute h and w indices from 2D block and thread indices
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds for spatial dimensions
    if (h_out >= out_h || w_out >= out_w) return;

    // Determine pooling window boundaries in input
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    // Pre-compute clamped boundaries
    int d_start_clamped = max(d_start, 0);
    int h_start_clamped = max(h_start, 0);
    int w_start_clamped = max(w_start, 0);
    int d_end_clamped = min(d_start + kernel_size, in_d);
    int h_end_clamped = min(h_start + kernel_size, in_h);
    int w_end_clamped = min(w_start + kernel_size, in_w);

    // Pre-compute base offset for current batch and channel
    int batch_channel_offset = (n * channels + c) * in_d;
    float sum = 0.0f;

    #pragma unroll 2
    for (int d = d_start_clamped; d < d_end_clamped; d++) {
        int d_offset = (batch_channel_offset + d) * in_h;
        #pragma unroll 2
        for (int h = h_start_clamped; h < h_end_clamped; h++) {
            // Compute row base pointer for efficient width iteration
            int row_offset = (d_offset + h) * in_w;
            #pragma unroll 4
            for (int w = w_start_clamped; w < w_end_clamped; w++) {
                sum += input[row_offset + w];
            }
        }
    }

    // For count_include_pad=True, division is by the full pooling volume
    int pool_volume = kernel_size * kernel_size * kernel_size;
    int output_idx = ((((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out);
    output[output_idx] = sum / static_cast<float>(pool_volume);
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Use 16x16 thread block for better occupancy
    dim3 block(8, 8, 1);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              batch_size * channels * out_d);

    avg_pool3d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with tuned block size");
}