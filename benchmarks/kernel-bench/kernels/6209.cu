#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constant memory for kernel parameters
__constant__ int kernel_data[4]; // Contains kernel_size, stride, padding, pool_volume

// Device function to decompose a linear index into 5D coordinates
__device__ void decompose_index(int index, int out_w, int out_h, int out_d, int channels,
                                int &n, int &c, int &d_out, int &h_out, int &w_out) {
    w_out = index % out_w;
    int tmp = index / out_w;
    h_out = tmp % out_h;
    tmp = tmp / out_h;
    d_out = tmp % out_d;
    tmp = tmp / out_d;
    c = tmp % channels;
    n = tmp / channels;
}

// Device function to compute the sum over a pooling window in the input using branchless operations
__device__ float compute_window_sum_branchless(const float* __restrict__ input,
                                               int n, int c,
                                               int d_out, int h_out, int w_out,
                                               int in_d, int in_h, int in_w,
                                               int channels) {
    // Retrieve the kernel parameters from constant memory
    int kernel_size = kernel_data[0];
    int stride = kernel_data[1];
    int padding = kernel_data[2];

    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    int d_end = d_start + kernel_size;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    int d0 = max(d_start, 0);
    int h0 = max(h_start, 0);
    int w0 = max(w_start, 0);

    int d1 = min(d_end, in_d);
    int h1 = min(h_end, in_h);
    int w1 = min(w_end, in_w);

    float sum = 0.0f;
    for (int d = d0; d < d1; ++d) {
        int base_d = ((n * channels + c) * in_d + d) * in_h * in_w;
        for (int h = h0; h < h1; ++h) {
            int base_h = base_d + h * in_w;
            for (int w = w0; w < w1; ++w) {
                sum += input[base_h + w];
            }
        }
    }
    return sum;
}

// Optimized kernel using constant memory for 3D average pooling
__global__ void avg_pool3d_constant_memory_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    // Retrieve pool volume from constant memory
    int pool_volume = kernel_data[3];

    while (index < total_elements) {
        int n, c, d_out, h_out, w_out;
        decompose_index(index, out_w, out_h, out_d, channels, n, c, d_out, h_out, w_out);

        float sum = compute_window_sum_branchless(input, n, c, d_out, h_out, w_out,
                                                  in_d, in_h, in_w, channels);
        output[index] = sum / static_cast<float>(pool_volume);

        index += blockDim.x * gridDim.x;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    // Ensure input is a 5D CUDA tensor
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Compute the output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Set the kernel parameters in constant memory
    int params[4] = {kernel_size, stride, padding, kernel_size * kernel_size * kernel_size};
    cudaMemcpyToSymbol(kernel_data, params, sizeof(int) * 4);

    avg_pool3d_constant_memory_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) - constant memory version");
}