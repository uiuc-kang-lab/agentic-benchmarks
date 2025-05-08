#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_forward_3dgrid(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {
    
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (d >= out_d || h >= out_h || w >= out_w) return;

    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            int d_start = d * stride - padding;
            int h_start = h * stride - padding;
            int w_start = w * stride - padding;

            int d_end = d_start + kernel_size;
            int h_end = h_start + kernel_size;
            int w_end = w_start + kernel_size;

            int d_start_clamped = max(d_start, 0);
            int h_start_clamped = max(h_start, 0);
            int w_start_clamped = max(w_start, 0);
            int d_end_clamped = min(d_end, in_d);
            int h_end_clamped = min(h_end, in_h);
            int w_end_clamped = min(w_end, in_w);

            float sum = 0.0f;
            for (int pd = d_start_clamped; pd < d_end_clamped; ++pd) {
                for (int ph = h_start_clamped; ph < h_end_clamped; ++ph) {
                    for (int pw = w_start_clamped; pw < w_end_clamped; ++pw) {
                        int input_idx = ((n * channels + c) * in_d + pd) * in_h * in_w + ph * in_w + pw;
                        sum += input[input_idx];
                    }
                }
            }

            int pool_volume = kernel_size * kernel_size * kernel_size;
            int output_idx = ((n * channels + c) * out_d + d) * out_h * out_w + h * out_w + w;
            output[output_idx] = sum / pool_volume;
        }
    }
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

    dim3 threads(8, 8, 4);  // 256 threads per block
    dim3 grids(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        (out_d + threads.z - 1) / threads.z
    );

    avg_pool3d_forward_3dgrid<<<grids, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling with 3D grid optimization");
}