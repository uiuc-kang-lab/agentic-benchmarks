#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

__global__ void avg_pool3d_forward_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int index = blockIdx.x * blockDim.x + threadIdx.x * 3;
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    while (index < total_elements) {
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

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
        int nc = n * channels + c;

        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
            int d_offset = nc * in_d + d;
            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                int h_offset = d_offset * in_h + h;
                int base_idx = h_offset * in_w;
                for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                    sum += input[base_idx + w];
                }
            }
        }

        int pool_volume = kernel_size * kernel_size * kernel_size;
        output[index] = sum / pool_volume;
        index += blockDim.x * gridDim.x;
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

    int out_d = (in_d + 2*padding - kernel_size)/stride + 1;
    int out_h = (in_h + 2*padding - kernel_size)/stride + 1;
    int out_w = (in_w + 2*padding - kernel_size)/stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    int total_elements = output.numel();

    dim3 blocks((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);

    avg_pool3d_forward_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling optimized with index calculation improvements");
}