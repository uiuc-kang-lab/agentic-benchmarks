#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_W = 32, int BLOCK_H = 8>
__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Shared memory to cache input tiles
    extern __shared__ float shared_input[];
    
    // Use blockIdx.z for (n, c, d_out) combined
    int idx = blockIdx.z;
    int d_out = idx % out_d;
    idx /= out_d;
    int c = idx % channels;
    int n = idx / channels;

    int h_out = blockIdx.y * BLOCK_H + threadIdx.y;
    int w_out = blockIdx.x * BLOCK_W + threadIdx.x;

    // Early exit if outside output bounds
    if (h_out >= out_h || w_out >= out_w) return;

    // Calculate input region needed for this thread block
    int h_in_start = blockIdx.y * BLOCK_H * stride - padding;
    int w_in_start = blockIdx.x * BLOCK_W * stride - padding;
    
    // Calculate shared memory dimensions including padding for pooling window
    int shared_h = BLOCK_H * stride + kernel_size;
    int shared_w = BLOCK_W * stride + kernel_size;

    // Load input data into shared memory
    int d_start = d_out * stride - padding;
    int d_end = min(d_start + kernel_size, in_d);
    d_start = max(0, d_start);

    for (int d = d_start; d < d_end; d++) {
        for (int h = threadIdx.y; h < shared_h; h += BLOCK_H) {
            int h_in = h_in_start + h;
            if (h_in < 0 || h_in >= in_h) continue;

            for (int w = threadIdx.x; w < shared_w; w += BLOCK_W) {
                int w_in = w_in_start + w;
                if (w_in < 0 || w_in >= in_w) continue;

                int input_idx = (((n * channels + c) * in_d + d) * in_h + h_in) * in_w + w_in;
                int shared_idx = ((d - d_start) * shared_h + h) * shared_w + w;
                shared_input[shared_idx] = input[input_idx];
            }
        }
    }

    __syncthreads();

    // Compute pooling window boundaries for this output element
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    float sum = 0.0f;
    
    // Compute pooling using shared memory
    for (int d = d_start; d < d_end; d++) {
        for (int h = max(0, h_start); h < min(h_start + kernel_size, in_h); h++) {
            for (int w = max(0, w_start); w < min(w_start + kernel_size, in_w); w++) {
                // Convert global coordinates to shared memory coordinates
                int h_shared = h - h_in_start;
                int w_shared = w - w_in_start;
                if (h_shared >= 0 && h_shared < shared_h && w_shared >= 0 && w_shared < shared_w) {
                    int shared_idx = ((d - d_start) * shared_h + h_shared) * shared_w + w_shared;
                    sum += shared_input[shared_idx];
                }
            }
        }
    }

    // Write output
    int pool_volume = kernel_size * kernel_size * kernel_size;
    int output_idx = ((((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out);
    output[output_idx] = sum / static_cast<float>(pool_volume);
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

    constexpr int BLOCK_W = 32;
    constexpr int BLOCK_H = 8;
    
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((out_w + BLOCK_W - 1) / BLOCK_W,
              (out_h + BLOCK_H - 1) / BLOCK_H,
              batch_size * channels * out_d);

    // Calculate shared memory size
    int shared_h = BLOCK_H * stride + kernel_size;
    int shared_w = BLOCK_W * stride + kernel_size;
    int shared_mem_size = kernel_size * shared_h * shared_w * sizeof(float);

    avg_pool3d_forward_kernel<BLOCK_W, BLOCK_H><<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with shared memory");
}