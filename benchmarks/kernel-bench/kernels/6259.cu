#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile sizes for shared memory tiling
#define TILE_D 4
#define TILE_H 4
#define TILE_W 4

__global__ void optimized_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Calculate (n, c) from blockIdx.x
    int bc = blockIdx.x;
    int n = bc / channels;
    int c = bc % channels;

    // Calculate tile indices
    int d_tile_idx = blockIdx.y;
    int h_tile_idx = blockIdx.z / ((out_w + TILE_W - 1) / TILE_W);
    int w_tile_idx = blockIdx.z % ((out_w + TILE_W - 1) / TILE_W);

    // Calculate output start indices
    int d_out_start = d_tile_idx * TILE_D;
    int h_out_start = h_tile_idx * TILE_H;
    int w_out_start = w_tile_idx * TILE_W;

    // Calculate shared memory dimensions
    int shared_d = ((TILE_D - 1) * stride + kernel_size);
    int shared_h = ((TILE_H - 1) * stride + kernel_size);
    int shared_w = ((TILE_W - 1) * stride + kernel_size);

    // Calculate input start indices
    int d_in_start = d_out_start * stride - padding;
    int h_in_start = h_out_start * stride - padding;
    int w_in_start = w_out_start * stride - padding;

    // Allocate shared memory
    extern __shared__ float shmem[];
    int shared_tile_size = shared_d * shared_h * shared_w;

    // Load input into shared memory
    int tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int idx = tid; idx < shared_tile_size; idx += block_threads) {
        int s_d = idx / (shared_h * shared_w);
        int rem = idx % (shared_h * shared_w);
        int s_h = rem / shared_w;
        int s_w = rem % shared_w;

        int in_d_idx = d_in_start + s_d;
        int in_h_idx = h_in_start + s_h;
        int in_w_idx = w_in_start + s_w;

        float val = 0.0f;
        if (in_d_idx >= 0 && in_d_idx < in_d &&
            in_h_idx >= 0 && in_h_idx < in_h &&
            in_w_idx >= 0 && in_w_idx < in_w) {
            int input_index = (((n * channels + c) * in_d + in_d_idx) * in_h + in_h_idx) * in_w + in_w_idx;
            val = input[input_index];
        }
        shmem[idx] = val;
    }

    __syncthreads();

    // Calculate output indices
    int t_w = threadIdx.x;
    int t_h = threadIdx.y;
    int t_d = threadIdx.z;

    if (t_w < TILE_W && t_h < TILE_H && t_d < TILE_D) {
        int d_out = d_out_start + t_d;
        int h_out = h_out_start + t_h;
        int w_out = w_out_start + t_w;

        if (d_out < out_d && h_out < out_h && w_out < out_w) {
            int in_d_pool = d_out * stride - padding;
            int in_h_pool = h_out * stride - padding;
            int in_w_pool = w_out * stride - padding;

            int s_d_start = in_d_pool - d_in_start;
            int s_h_start = in_h_pool - h_in_start;
            int s_w_start = in_w_pool - w_in_start;

            float sum = 0.0f;
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int s_d_idx = s_d_start + kd;
                        int s_h_idx = s_h_start + kh;
                        int s_w_idx = s_w_start + kw;
                        if (s_d_idx >= 0 && s_d_idx < shared_d &&
                            s_h_idx >= 0 && s_h_idx < shared_h &&
                            s_w_idx >= 0 && s_w_idx < shared_w) {
                            int shmem_idx = (s_d_idx * shared_h * shared_w) + (s_h_idx * shared_w) + s_w_idx;
                            sum += shmem[shmem_idx];
                        }
                    }
                }
            }

            float avg = sum / (kernel_size * kernel_size * kernel_size);
            int out_index = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
            output[out_index] = avg;
        }
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional (N, C, D, H, W)");
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

    dim3 grid(batch_size * channels, (out_d + TILE_D - 1) / TILE_D, (out_h + TILE_H - 1) / TILE_H * (out_w + TILE_W - 1) / TILE_W);
    dim3 block(TILE_W, TILE_H, TILE_D);

    int shared_d = ((TILE_D - 1) * stride + kernel_size);
    int shared_h = ((TILE_H - 1) * stride + kernel_size);
    int shared_w = ((TILE_W - 1) * stride + kernel_size);
    size_t shared_mem_size = shared_d * shared_h * shared_w * sizeof(float);

    optimized_avg_pool3d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D Average Pooling forward (CUDA)");
}