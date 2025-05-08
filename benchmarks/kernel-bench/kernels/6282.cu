#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int VEC_SIZE = 4;

__global__ void avg_pool3d_forward_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * out_d * out_h * out_w;
    const float inv_volume = 1.0f / (kernel_size * kernel_size * kernel_size);

    if (index < total_elements) {
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp /= out_h;
        int d_out = tmp % out_d;
        const int c = (tmp / out_d) % channels;
        const int n = (tmp / out_d) / channels;

        const int d_start = max(d_out * stride - padding, 0);
        const int d_end = min(d_start + kernel_size, in_d);
        const int h_start = max(h_out * stride - padding, 0);
        const int h_end = min(h_start + kernel_size, in_h);
        const int w_start = max(w_out * stride - padding, 0);
        const int w_end = min(w_start + kernel_size, in_w);

        float sum = 0.0f;
        for (int d = d_start; d < d_end; ++d) {
            for (int h = h_start; h < h_end; ++h) {
                int input_idx = (((n * channels + c) * in_d + d) * in_h + h) * in_w + w_start;
                int w_remaining = w_end - w_start;
                
                // Vectorized load for aligned memory access
                if ((input_idx % VEC_SIZE) == 0 && w_remaining >= VEC_SIZE) {
                    const float4* vec_ptr = reinterpret_cast<const float4*>(input + input_idx);
                    for (; w_remaining >= VEC_SIZE; w_remaining -= VEC_SIZE) {
                        float4 vec_val = *vec_ptr++;
                        sum += vec_val.x + vec_val.y + vec_val.z + vec_val.w;
                    }
                    input_idx += (w_end - w_start - w_remaining);
                }

                // Process remaining elements
                for (int w = 0; w < w_remaining; ++w) {
                    sum += input[input_idx++];
                }
            }
        }

        output[index] = sum * inv_volume;
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);

    const int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    const int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());
    const int total_elements = output.numel();
    const int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    avg_pool3d_forward_optimized<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D Average Pooling forward (CUDA)");
}