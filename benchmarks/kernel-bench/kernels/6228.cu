#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 3D Average Pooling Kernel using shared memory
__global__ void avg_pool3d_shared_mem_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int grid_stride = blockDim.x * gridDim.x;
    const int total_elements = batch_size * channels * out_d * out_h * out_w;
    const float inv_pool_volume = 1.0f / (kernel_size * kernel_size * kernel_size);

    // Grid-stride loop to cover all output elements
    for (; index < total_elements; index += grid_stride) {
        int tmp = index;
        int w_out = tmp % out_w;
        tmp /= out_w;
        int h_out = tmp % out_h;
        tmp /= out_h;
        int d_out = tmp % out_d;
        tmp /= out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        int d0 = max(d_start, 0);
        int h0 = max(h_start, 0);
        int w0 = max(w_start, 0);
        int d1 = min(d_start + kernel_size, in_d);
        int h1 = min(h_start + kernel_size, in_h);
        int w1 = min(w_start + kernel_size, in_w);

        float sum = 0.0f;
        int base_nc = (n * channels + c) * in_d * in_h * in_w;

        for (int d = d0; d < d1; ++d) {
            int base_d = base_nc + d * in_h * in_w;
            for (int h = h0; h < h1; ++h) {
                int base_h = base_d + h * in_w;
                for (int w = w0; w < w1; ++w) {
                    int input_idx = base_h + w;
                    sum += input[input_idx];
                }
            }
        }

        shared_mem[tid] = sum;
        __syncthreads();

        if (tid == 0) {
            float block_sum = 0.0f;
            for (int i = 0; i < blockDim.x; ++i) {
                block_sum += shared_mem[i];
            }
            output[index] = block_sum * inv_pool_volume;
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

    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    int shared_mem_size = threads * sizeof(float);

    avg_pool3d_shared_mem_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) using shared memory");
}
