#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D average pooling (count_include_pad=True) using stride loops
// to allow each thread to process multiple output elements if total workload exceeds thread count.
__global__ void avg_pool3d_forward_kernel_stride(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int pool_stride, int padding) {

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int grid_stride = blockDim.x * gridDim.x;

    // Stride loop: each thread processes multiple elements spaced by grid_stride
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_elements; index += grid_stride) {
        // Decompose the linear index into (n, c, d_out, h_out, w_out)
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;
        
        // Compute the start indices of the pooling window (with padding taken into account)
        int d_start = d_out * pool_stride - padding;
        int h_start = h_out * pool_stride - padding;
        int w_start = w_out * pool_stride - padding;
        
        // Compute the end boundaries for the pooling window
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;
        
        // Clamp window boundaries to stay within the input dimensions
        int d_start_clamped = (d_start < 0) ? 0 : d_start;
        int h_start_clamped = (h_start < 0) ? 0 : h_start;
        int w_start_clamped = (w_start < 0) ? 0 : w_start;
        int d_end_clamped = (d_end > in_d) ? in_d : d_end;
        int h_end_clamped = (h_end > in_h) ? in_h : h_end;
        int w_end_clamped = (w_end > in_w) ? in_w : w_end;
        
        float sum = 0.0f;
        // Accumulate sum over the pooling window
        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                    int input_idx = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
                    sum += input[input_idx];
                }
            }
        }
        
        // For count_include_pad=True, always divide by the full volume
        int pool_volume = kernel_size * kernel_size * kernel_size;
        output[index] = sum / static_cast<float>(pool_volume);
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
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

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Enforce the CUDA grid limit on the x-dimension
    if (blocks > 65535) blocks = 65535;

    avg_pool3d_forward_kernel_stride<<<blocks, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with stride loops");
}
