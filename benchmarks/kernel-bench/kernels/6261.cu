#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined optimized 3D average pooling kernel
__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Map grid to cover (n, c) and the d dimension using a 3D thread block
    int d_tile = (out_d + blockDim.z - 1) / blockDim.z;
    int nc = blockIdx.z / d_tile;
    int tile_idx = blockIdx.z % d_tile;
    int n = nc / channels;
    int c = nc % channels;
    int d_out = tile_idx * blockDim.z + threadIdx.z;
    if (d_out >= out_d) return;

    // Compute output spatial coordinates
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (h_out >= out_h || w_out >= out_w) return;

    // Compute pooling window start indices (considering padding)
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Clamp pooling window boundaries to input dimensions
    int d_start_clamped = (d_start < 0) ? 0 : d_start;
    int h_start_clamped = (h_start < 0) ? 0 : h_start;
    int w_start_clamped = (w_start < 0) ? 0 : w_start;

    int d_end_clamped = (d_start + kernel_size > in_d) ? in_d : (d_start + kernel_size);
    int h_end_clamped = (h_start + kernel_size > in_h) ? in_h : (h_start + kernel_size);
    int w_end_clamped = (w_start + kernel_size > in_w) ? in_w : (w_start + kernel_size);

    float sum = 0.0f;

    // Precompute the base offset for the current (n, c) slice
    int input_channel_offset = ((n * channels + c) * in_d * in_h * in_w);

    // Loop over the 3D pooling window using efficient pointer arithmetic
    for (int d = d_start_clamped; d < d_end_clamped; d++) {
        for (int h = h_start_clamped; h < h_end_clamped; h++) {
            // Compute the row base pointer for the current d and h
            int row_index = input_channel_offset + (d * in_h + h) * in_w;
            int row_start_idx = row_index + w_start_clamped;
            int row_length = w_end_clamped - w_start_clamped;
            #pragma unroll
            for (int i = 0; i < row_length; i++) {
                sum += input[row_start_idx + i];
            }
        }
    }

    // For count_include_pad=True, the divisor is always the full pooling volume
    int pool_volume = kernel_size * kernel_size * kernel_size;
    int output_index = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
    output[output_index] = sum / static_cast<float>(pool_volume);
}

// Launcher function for PyTorch
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

    // Use block dimensions that favor coalesced memory accesses along the width dimension
    dim3 block(32, 8, 1); // 32 threads along width (match warp size) and 8 along height
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with combined optimizations");
}
