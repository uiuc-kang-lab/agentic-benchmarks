#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function: Convert linear index to 5D coordinates (n, c, d_out, h_out, w_out)
__device__ __forceinline__ void calculate_indices(
    int index, int out_w, int out_h, int out_d, int channels,
    int &n, int &c, int &d_out, int &h_out, int &w_out) {

    w_out = index % out_w;
    int tmp = index / out_w;
    h_out = tmp % out_h;
    tmp /= out_h;
    d_out = tmp % out_d;
    tmp /= out_d;
    c = tmp % channels;
    n = tmp / channels;
}

// Device function: Compute clamped window boundaries using branchless max/min operations
__device__ __forceinline__ void get_window_bounds(
    int d_out, int h_out, int w_out,
    int stride, int padding, int kernel_size,
    int in_d, int in_h, int in_w,
    int &d0, int &d1, int &h0, int &h1, int &w0, int &w1) {

    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    d0 = max(d_start, 0);
    h0 = max(h_start, 0);
    w0 = max(w_start, 0);

    d1 = min(d_start + kernel_size, in_d);
    h1 = min(h_start + kernel_size, in_h);
    w1 = min(w_start + kernel_size, in_w);
}

// Device function: Compute the sum over the pooling window with loop unrolling
__device__ __forceinline__ float compute_window_sum(
    const float* __restrict__ input,
    int n, int c,
    int d0, int d1, int h0, int h1, int w0, int w1,
    int in_d, int in_h, int in_w, int channels) {

    float sum = 0.0f;
    // Compute the base index for (n, c) in a contiguous 5D tensor
    int base_nc = (n * channels + c) * in_d * in_h * in_w;

    #pragma unroll 2
    for (int d = d0; d < d1; ++d) {
        int base_d = base_nc + d * in_h * in_w;
        #pragma unroll 2
        for (int h = h0; h < h1; ++h) {
            int base_h = base_d + h * in_w;
            #pragma unroll 4
            for (int w = w0; w < w1; ++w) {
                sum += input[base_h + w];
            }
        }
    }
    return sum;
}

// Combined kernel that integrates modularity, branchless window boundary computation,
// and loop unrolling for inner loops
__global__ void avg_pool3d_combined_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    float inv_kernel_vol = 1.0f / (kernel_size * kernel_size * kernel_size);
    int grid_stride = blockDim.x * gridDim.x;

    while (idx < total_elements) {
        int n, c, d_out, h_out, w_out;
        calculate_indices(idx, out_w, out_h, out_d, channels,
                          n, c, d_out, h_out, w_out);

        int d0, d1, h0, h1, w0, w1;
        get_window_bounds(d_out, h_out, w_out,
                          stride, padding, kernel_size,
                          in_d, in_h, in_w,
                          d0, d1, h0, h1, w0, w1);

        float sum = compute_window_sum(input, n, c,
                                         d0, d1, h0, h1, w0, w1,
                                         in_d, in_h, in_w, channels);
        output[idx] = sum * inv_kernel_vol;

        idx += grid_stride;
    }
}

// Interface function to be called from PyTorch
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

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    avg_pool3d_combined_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Combined 3D Average Pooling forward (CUDA)");
}
