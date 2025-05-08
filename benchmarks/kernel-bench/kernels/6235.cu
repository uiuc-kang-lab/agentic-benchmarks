#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D average pooling using fixed iteration loops to minimize warp divergence
__global__ void avg_pool3d_forward_kernel_minimize_divergence(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    // Each thread processes multiple output elements
    while (index < total_elements) {
        // Decompose the linear index into (n, c, d_out, h_out, w_out)
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute the starting coordinate of the pooling window
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        float sum = 0.0f;
        // Use fixed triple nested loops over the kernel window
        // This ensures every thread performs the same number of iterations,
        // reducing warp divergence due to variable loop bounds.
        for (int i = 0; i < kernel_size; i++) {
            int d = d_start + i;
            // Pre-calculate validity flag for d dimension
            bool valid_d = (d >= 0) && (d < in_d);
            for (int j = 0; j < kernel_size; j++) {
                int h = h_start + j;
                bool valid_h = (h >= 0) && (h < in_h);
                for (int k = 0; k < kernel_size; k++) {
                    int w = w_start + k;
                    bool valid_w = (w >= 0) && (w < in_w);
                    // Instead of varying loop bounds, always iterate and use a conditional
                    // to add only valid input elements. Out-of-bound positions contribute 0.
                    if (valid_d && valid_h && valid_w) {
                        int input_index = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
                        sum += input[input_index];
                    }
                }
            }
        }
        // For count_include_pad=True, always divide by full kernel volume
        int pool_volume = kernel_size * kernel_size * kernel_size;
        output[index] = sum / static_cast<float>(pool_volume);

        index += blockDim.x * gridDim.x;
    }
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

    // Compute output dimensions using the pooling formula:
    // out_dim = floor((in_dim + 2*padding - kernel_size) / stride) + 1
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor with the same options as input.
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    avg_pool3d_forward_kernel_minimize_divergence<<<blocks, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with minimized divergence");
}
