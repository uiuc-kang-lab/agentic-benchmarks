#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for shared memory
#define TILE_D 8
#define TILE_H 8
#define TILE_W 16

__global__ void avg_pool3d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    extern __shared__ float shared_input[];

    // Calculate thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    // Calculate output indices
    const int w_out = blockIdx.x * blockDim.x + tx;
    const int h_out = blockIdx.y * blockDim.y + ty;
    const int d_out = blockIdx.z * blockDim.z + tz;
    
    // Calculate batch and channel indices
    const int c = blockIdx.w % channels;
    const int n = blockIdx.w / channels;

    if (w_out >= out_w || h_out >= out_h || d_out >= out_d) return;

    // Calculate input window boundaries
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    const int d_end = min(d_start + kernel_size, in_d);
    const int h_end = min(h_start + kernel_size, in_h);
    const int w_end = min(w_start + kernel_size, in_w);
    
    const int d_start_valid = max(d_start, 0);
    const int h_start_valid = max(h_start, 0);
    const int w_start_valid = max(w_start, 0);

    // Calculate shared memory indices
    const int tile_size = TILE_D * TILE_H * TILE_W;
    const int shared_idx_base = (tz * TILE_H + ty) * TILE_W + tx;

    float sum = 0.0f;
    const int pool_volume = kernel_size * kernel_size * kernel_size;
    
    // Process input in tiles
    for (int d = d_start_valid; d < d_end; d += TILE_D) {
        for (int h = h_start_valid; h < h_end; h += TILE_H) {
            for (int w = w_start_valid; w < w_end; w += TILE_W) {
                // Load input tile into shared memory
                const int d_limit = min(d + TILE_D, d_end);
                const int h_limit = min(h + TILE_H, h_end);
                const int w_limit = min(w + TILE_W, w_end);

                __syncthreads();

                // Collaborative loading of tile into shared memory
                for (int td = d; td < d_limit; td++) {
                    for (int th = h; th < h_limit; th++) {
                        for (int tw = w; tw < w_limit; tw++) {
                            if (tx + tw - w < TILE_W && ty + th - h < TILE_H && tz + td - d < TILE_D) {
                                const int input_idx = ((n * channels + c) * in_d + td) * in_h * in_w +
                                                    th * in_w + tw;
                                const int shared_idx = ((td - d) * TILE_H + (th - h)) * TILE_W + (tw - w);
                                shared_input[shared_idx] = input[input_idx];
                            }
                        }
                    }
                }

                __syncthreads();

                // Process the tile
                for (int td = 0; td < d_limit - d; td++) {
                    for (int th = 0; th < h_limit - h; th++) {
                        for (int tw = 0; tw < w_limit - w; tw++) {
                            const int shared_idx = (td * TILE_H + th) * TILE_W + tw;
                            sum += shared_input[shared_idx];
                        }
                    }
                }
            }
        }
    }

    // Write output
    if (w_out < out_w && h_out < out_h && d_out < out_d) {
        const int output_idx = ((n * channels + c) * out_d + d_out) * out_h * out_w +
                             h_out * out_w + w_out;
        output[output_idx] = sum / static_cast<float>(pool_volume);
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

    // Define block and grid dimensions
    dim3 threadsPerBlock(8, 8, 4);
    dim3 numBlocks(
        (out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_h + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (out_d + threadsPerBlock.z - 1) / threadsPerBlock.z,
        batch_size * channels
    );

    // Calculate shared memory size
    int shared_memory_size = TILE_D * TILE_H * TILE_W * sizeof(float);

    avg_pool3d_shared_kernel<<<numBlocks, threadsPerBlock, shared_memory_size>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with shared memory");
}