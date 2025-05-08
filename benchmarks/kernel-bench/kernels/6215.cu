#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for 3D average pooling with memory coalescing
// Memory accesses are optimized for coalescing by ensuring threads access consecutive memory locations.

__global__ void avg_pool3d_coalesced_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_size = blockDim.x * gridDim.x;
    int kernelVolume = kernel_size * kernel_size * kernel_size;

    for (int index = idx; index < batch_size * channels * out_d * out_h * out_w; index += stride_size) {
        // Decompose the linear index into (n, c, d_out, h_out, w_out)
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Pre-calculate start and end indices
        int d_start = max(d_out * stride - padding, 0);
        int h_start = max(h_out * stride - padding, 0);
        int w_start = max(w_out * stride - padding, 0);

        int d_end = min(d_start + kernel_size, in_d);
        int h_end = min(h_start + kernel_size, in_h);
        int w_end = min(w_start + kernel_size, in_w);

        float sum = 0.0f;
        // Loop over the pooling window
        for (int d = d_start; d < d_end; ++d) {
            int base_d = ((n * channels + c) * in_d + d) * in_h * in_w;
            for (int h = h_start; h < h_end; ++h) {
                int base_h = base_d + h * in_w;
                for (int w = w_start; w < w_end; ++w) {
                    sum += input[base_h + w];
                }
            }
        }

        // Divide by full kernel volume
        output[idx] = sum / static_cast<float>(kernelVolume);
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

    avg_pool3d_coalesced_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) - coalesced version");
}
