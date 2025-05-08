#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>


constexpr int BLOCK_SIZE = 128;

// CUDA kernel for 3D average pooling (count_include_pad=True) using __ldg() for read-only loads
// and attempting 128-bit aligned vectorized loads for the innermost loop.
__global__ void avg_pool3d_forward_kernel_ldg(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int grid_stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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

        // Compute starting indices of the pooling window
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        
        int d_end = d_start + kernel_size;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;

        // Clamp window boundaries to input dimensions
        int d_start_clamped = (d_start < 0) ? 0 : d_start;
        int h_start_clamped = (h_start < 0) ? 0 : h_start;
        int w_start_clamped = (w_start < 0) ? 0 : w_start;
        int d_end_clamped = (d_end > in_d) ? in_d : d_end;
        int h_end_clamped = (h_end > in_h) ? in_h : h_end;
        int w_end_clamped = (w_end > in_w) ? in_w : w_end;

        float sum = 0.0f;
        // Calculate the base channel index
        int nc = n * channels + c;

        // Loop over the pooling window in d and h dimensions; optimize the w loop with vectorized loads
        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
            // Compute partial index for d and channel
            int d_offset = ((nc * in_d) + d) * in_h; 
            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                int base_idx = (d_offset + h) * in_w;  // starting index for this row in the w dimension
                int width = w_end_clamped - w_start_clamped;

                // Check if the address is 128-bit aligned and vectorization is possible
                uintptr_t addr = (uintptr_t)(&input[base_idx + w_start_clamped]);
                if (((addr & 0xF) == 0) && (width >= 4)) {
                    int vec_steps = width / 4;
                    int rem = width % 4;
                    const float4* vptr = reinterpret_cast<const float4*>(&input[base_idx + w_start_clamped]);
                    for (int i = 0; i < vec_steps; ++i) {
                        float4 vec = __ldg(vptr + i);
                        sum += vec.x + vec.y + vec.z + vec.w;
                    }
                    int start_rem = w_start_clamped + vec_steps * 4;
                    for (int w = start_rem; w < w_end_clamped; ++w) {
                        sum += __ldg(&input[base_idx + w]);
                    }
                } else {
                    for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                        sum += __ldg(&input[base_idx + w]);
                    }
                }
            }
        }

        // For count_include_pad=True, always divide by the full kernel volume
        int pool_volume = kernel_size * kernel_size * kernel_size;
        output[index] = sum / static_cast<float>(pool_volume);

        index += grid_stride;
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
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    int threads = BLOCK_SIZE;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    avg_pool3d_forward_kernel_ldg<<<blocks, threads>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward using __ldg() and 128-bit aligned loads");
}
