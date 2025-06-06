#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 3D Average Pooling Kernel that combines efficient index decomposition, branchless window bounds, and loop unrolling
__global__ void avg_pool3d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    const int total_elements = batch_size * channels * out_d * out_h * out_w;
    const float inv_pool_volume = 1.0f / (kernel_size * kernel_size * kernel_size);

    // Grid-stride loop to cover all output elements
    for (; index < total_elements; index += grid_stride) {
        // Inline decompose the linear index into 5D coordinates: n, c, d_out, h_out, w_out
        int tmp = index;
        int w_out = tmp % out_w;
        tmp /= out_w;
        int h_out = tmp % out_h;
        tmp /= out_h;
        int d_out = tmp % out_d;
        tmp /= out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute the pooling window boundaries using branchless clamping
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        int d0 = d_start < 0 ? 0 : d_start;
        int h0 = h_start < 0 ? 0 : h_start;
        int w0 = w_start < 0 ? 0 : w_start;
        int d1 = (d_start + kernel_size > in_d) ? in_d : d_start + kernel_size;
        int h1 = (h_start + kernel_size > in_h) ? in_h : h_start + kernel_size;
        int w1 = (w_start + kernel_size > in_w) ? in_w : w_start + kernel_size;

        // Efficient sum computation using pointer arithmetic and unrolling for the innermost loop
        float sum = 0.0f;
        // Precompute the base offset for the current (n, c) slice
        int base_nc = (n * channels + c) * in_d * in_h * in_w;
        for (int d = d0; d < d1; ++d) {
            int base_d = base_nc + d * in_h * in_w;
            for (int h = h0; h < h1; ++h) {
                int base_h = base_d + h * in_w;
                #pragma unroll
                for (int w = w0; w < w1; ++w) {
                    sum += input[base_h + w];
                }
            }
        }
        output[index] = sum * inv_pool_volume;
    }
}

// Host function that prepares and launches the kernel
at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    avg_pool3d_optimized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) optimized kernel");
}
