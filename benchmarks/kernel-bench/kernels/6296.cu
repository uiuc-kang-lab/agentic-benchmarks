#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Structure to hold pooling parameters in constant memory
struct PoolParams {
    int kernel_size;
    int stride;
    int padding;
    int in_d;
    int in_h;
    int in_w;
    int out_d;
    int out_h;
    int out_w;
};

// Declare pooling parameters in constant memory
__constant__ PoolParams d_params;

// CUDA kernel using constant memory for read-only pooling parameters
__global__ void avg_pool3d_forward_const(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels) {

    int total_elements = batch_size * channels * d_params.out_d * d_params.out_h * d_params.out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    
    while (idx < total_elements) {
        // Decompose idx into (n, c, d_out, h_out, w_out)
        int w_out = idx % d_params.out_w;
        int tmp = idx / d_params.out_w;
        int h_out = tmp % d_params.out_h;
        tmp = tmp / d_params.out_h;
        int d_out = tmp % d_params.out_d;
        tmp = tmp / d_params.out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute pooling window start indices
        int d_start = d_out * d_params.stride - d_params.padding;
        int h_start = h_out * d_params.stride - d_params.padding;
        int w_start = w_out * d_params.stride - d_params.padding;

        // Compute pooling window end indices
        int d_end = d_start + d_params.kernel_size;
        int h_end = h_start + d_params.kernel_size;
        int w_end = w_start + d_params.kernel_size;

        // Clamp window boundaries to valid input dimensions
        int d_start_clamped = (d_start < 0) ? 0 : d_start;
        int h_start_clamped = (h_start < 0) ? 0 : h_start;
        int w_start_clamped = (w_start < 0) ? 0 : w_start;
        int d_end_clamped = (d_end > d_params.in_d) ? d_params.in_d : d_end;
        int h_end_clamped = (h_end > d_params.in_h) ? d_params.in_h : h_end;
        int w_end_clamped = (w_end > d_params.in_w) ? d_params.in_w : w_end;

        float sum = 0.0f;
        // Compute base index for input access: index layout is [N, C, D, H, W]
        int nc = n * channels + c;
        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
            int d_offset = nc * d_params.in_d + d;
            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                int h_offset = d_offset * d_params.in_h + h;
                int base_idx = h_offset * d_params.in_w;
                for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                    sum += input[base_idx + w];
                }
            }
        }
        
        int pool_volume = d_params.kernel_size * d_params.kernel_size * d_params.kernel_size;
        output[idx] = sum / static_cast<float>(pool_volume);
        
        idx += grid_stride;
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

    // Compute output dimensions using pooling formula
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Prepare host pooling parameters structure
    PoolParams h_params;
    h_params.kernel_size = kernel_size;
    h_params.stride = stride;
    h_params.padding = padding;
    h_params.in_d = in_d;
    h_params.in_h = in_h;
    h_params.in_w = in_w;
    h_params.out_d = out_d;
    h_params.out_h = out_h;
    h_params.out_w = out_w;

    // Copy the parameters to constant memory
    cudaMemcpyToSymbol(d_params, &h_params, sizeof(PoolParams));

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    int total_elements = output.numel();

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // Ensure grid limit is met

    avg_pool3d_forward_const<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward with constant memory optimization (CUDA)");
}
