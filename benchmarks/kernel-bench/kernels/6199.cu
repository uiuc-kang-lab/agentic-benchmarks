#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D average pooling with coalesced memory accesses
__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // 3D thread block for coalesced memory access
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.z * blockDim.z + threadIdx.z;

    if (d_out >= out_d || h_out >= out_h || w_out >= out_w) return;

    // Calculate n and c using blockIdx and threadIdx.z for batches and channels
    int n = blockIdx.z / channels;
    int c = blockIdx.z % channels;

    // Compute the top-left-front corner of the pooling window
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Clamp window boundaries
    int d_start_clamped = max(d_start, 0);
    int h_start_clamped = max(h_start, 0);
    int w_start_clamped = max(w_start, 0);
    int d_end_clamped = min(d_start + kernel_size, in_d);
    int h_end_clamped = min(h_start + kernel_size, in_h);
    int w_end_clamped = min(w_start + kernel_size, in_w);

    float sum = 0.0f;
    for (int d = d_start_clamped; d < d_end_clamped; ++d) {
        for (int h = h_start_clamped; h < h_end_clamped; ++h) {
            for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                int input_index = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w;
                sum += input[input_index];
            }
        }
    }

    int pool_volume = kernel_size * kernel_size * kernel_size;
    int output_index = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
    output[output_index] = sum / static_cast<float>(pool_volume);
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    dim3 threads(8, 8, 4); // 3D thread block for coalesced access
    dim3 grids(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        (out_d * batch_size * channels + threads.z - 1) / threads.z
    );

    avg_pool3d_forward_kernel<<<grids, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward with coalesced memory access (CUDA)");
}
