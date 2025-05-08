#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D average pooling using grid-stride loops for large workloads
__global__ void avg_pool3d_stride_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch_size * channels * out_d * out_h * out_w;
    int stride_loop = blockDim.x * gridDim.x;
    int kernelVolume = kernel_size * kernel_size * kernel_size;

    // Use stride loops to cover all output elements
    for (int index = idx; index < total; index += stride_loop) {
        // Decompose linear index into (n, c, d_out, h_out, w_out)
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute the starting indices for the pooling window
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        
        // Compute the ending indices
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        // Clamp boundaries using branchless ternary operators
        int d0 = d_start < 0 ? 0 : d_start;
        int h0 = h_start < 0 ? 0 : h_start;
        int w0 = w_start < 0 ? 0 : w_start;
        int d1 = d_end > in_d ? in_d : d_end;
        int h1 = h_end > in_h ? in_h : h_end;
        int w1 = w_end > in_w ? in_w : w_end;

        float sum = 0.0f;
        // Loop over the pooling window
        for (int d = d0; d < d1; ++d) {
            int base_d = ((n * channels + c) * in_d + d) * in_h * in_w;
            for (int h = h0; h < h1; ++h) {
                int base_h = base_d + h * in_w;
                for (int w = w0; w < w1; ++w) {
                    sum += input[base_h + w];
                }
            }
        }
        
        // Always divide by full kernel volume (count_include_pad = true)
        output[index] = sum / static_cast<float>(kernelVolume);
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
    
    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    avg_pool3d_stride_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) - stride loop version");
}
