#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a 3D grid layout to ensure that threads within a warp access contiguous
// output positions along the width dimension, which in turn aligns their input accesses for
// the pooling window. For each output element, we precompute the row base pointer for the
// input, so that the inner loop over the width of the pooling window performs coalesced
// global memory reads.

__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Use blockIdx.z for (n, c, d_out) combined.
    int idx = blockIdx.z;
    int d_out = idx % out_d;
    idx /= out_d;
    int c = idx % channels;
    int n = idx / channels;

    // Compute h and w indices from 2D block (gridDim.x, gridDim.y) and threadIdx
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds for spatial dimensions
    if (h_out >= out_h || w_out >= out_w) return;

    // Determine pooling window boundaries in input
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    int d_end = d_start + kernel_size;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    // Clamp pooling window to input boundaries
    int d_start_clamped = d_start < 0 ? 0 : d_start;
    int h_start_clamped = h_start < 0 ? 0 : h_start;
    int w_start_clamped = w_start < 0 ? 0 : w_start;
    int d_end_clamped = d_end > in_d ? in_d : d_end;
    int h_end_clamped = h_end > in_h ? in_h : h_end;
    int w_end_clamped = w_end > in_w ? in_w : w_end;

    float sum = 0.0f;

    // Loop over the 3D pooling window. For each row, use pointer arithmetic to load
    // consecutive elements from global memory, which benefits from coalescing.
    for (int d = d_start_clamped; d < d_end_clamped; d++) {
        for (int h = h_start_clamped; h < h_end_clamped; h++) {
            // Compute the base pointer for the current row in the input tensor
            int row_start_idx = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w_start_clamped;
            int row_length = w_end_clamped - w_start_clamped;
            for (int offset = 0; offset < row_length; offset++) {
                sum += input[row_start_idx + offset];
            }
        }
    }

    // For count_include_pad=True, division is by the full pooling volume
    int pool_volume = kernel_size * kernel_size * kernel_size;
    int output_idx = ((((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out);
    output[output_idx] = sum / static_cast<float>(pool_volume);
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    // Check that input is a 5D CUDA tensor
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Calculate output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Define block dimensions to promote coalescing in the width dimension
    dim3 block(16, 16, 1);
    // grid.x and grid.y cover the spatial dimensions (w and h), grid.z covers (n, c, d_out)
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              batch_size * channels * out_d);

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    avg_pool3d_forward_kernel<<<grid, block>>>(
        input_ptr, output_ptr,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with coalesced memory accesses");
}
