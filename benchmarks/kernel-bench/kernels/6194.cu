#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with warp-level reduction: each warp computes one output element
__global__ void avg_pool3d_forward_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Each warp computes one output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    int total_out = batch_size * channels * out_d * out_h * out_w;
    if (warp_id >= total_out) return;

    // Decode the linear warp_id into (n, c, d_out, h_out, w_out)
    int w_out = warp_id % out_w;
    int tmp = warp_id / out_w;
    int h_out = tmp % out_h;
    tmp = tmp / out_h;
    int d_out = tmp % out_d;
    tmp = tmp / out_d;
    int c = tmp % channels;
    int n = tmp / channels;

    // Compute the starting index of the pooling window with padding
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    int pool_volume = kernel_size * kernel_size * kernel_size;
    float thread_sum = 0.0f;

    // Each thread in the warp processes a subset of the pooling window elements
    // Using a flat index i into the pooling volume
    for (int i = lane; i < pool_volume; i += warpSize) {
        int d_offset = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int h_offset = rem / kernel_size;
        int w_offset = rem % kernel_size;

        int d = d_start + d_offset;
        int h = h_start + h_offset;
        int w = w_start + w_offset;

        // Only add valid input values (simulate zero-padding for out-of-bound indices)
        if (d >= 0 && d < in_d && h >= 0 && h < in_h && w >= 0 && w < in_w) {
            int input_index = ((((n * channels + c) * in_d + d) * in_h) + h) * in_w + w;
            thread_sum += input[input_index];
        }
    }

    // Perform warp-level reduction using __shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Lane 0 writes the computed average to the output
    if (lane == 0) {
        output[warp_id] = thread_sum / static_cast<float>(pool_volume);
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

    // Allocate the output tensor
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_out = batch_size * channels * out_d * out_h * out_w;
    // Each warp computes one output element; total threads = total_out * warpSize
    int total_threads = total_out * warpSize;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    avg_pool3d_forward_warp_kernel<<<blocks, threads>>>(input_ptr, output_ptr,
                                                         batch_size, channels,
                                                         in_d, in_h, in_w,
                                                         out_d, out_h, out_w,
                                                         kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward with warp-level reduction (CUDA)");
}
