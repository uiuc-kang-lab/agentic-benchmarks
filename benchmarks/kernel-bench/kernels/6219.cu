#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define shared memory tile dimensions
#define TILE_SIZE 8
#define THREADS_PER_BLOCK 256

__global__ void avg_pool3d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    __shared__ float shared_input[TILE_SIZE][TILE_SIZE][TILE_SIZE];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    
    while (index < total_elements) {
        // Decompose linear index
        int w_out = index % out_w;
        int tmp = index / out_w;
        int h_out = tmp % out_h;
        tmp = tmp / out_h;
        int d_out = tmp % out_d;
        tmp = tmp / out_d;
        int c = tmp % channels;
        int n = tmp / channels;

        // Calculate input boundaries
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        float sum = 0.0f;

        // Process input in tiles
        for (int tile_d = 0; tile_d < kernel_size; tile_d += TILE_SIZE) {
            for (int tile_h = 0; tile_h < kernel_size; tile_h += TILE_SIZE) {
                for (int tile_w = 0; tile_w < kernel_size; tile_w += TILE_SIZE) {
                    
                    // Load tile into shared memory
                    if (threadIdx.x < TILE_SIZE * TILE_SIZE * TILE_SIZE) {
                        int local_idx = threadIdx.x;
                        int local_d = local_idx / (TILE_SIZE * TILE_SIZE);
                        int local_tmp = local_idx % (TILE_SIZE * TILE_SIZE);
                        int local_h = local_tmp / TILE_SIZE;
                        int local_w = local_tmp % TILE_SIZE;

                        int d = d_start + tile_d + local_d;
                        int h = h_start + tile_h + local_h;
                        int w = w_start + tile_w + local_w;

                        if (d >= 0 && d < in_d && h >= 0 && h < in_h && w >= 0 && w < in_w) {
                            shared_input[local_d][local_h][local_w] = 
                                input[((n * channels + c) * in_d + d) * in_h * in_w + h * in_w + w];
                        } else {
                            shared_input[local_d][local_h][local_w] = 0.0f;
                        }
                    }
                    
                    __syncthreads();

                    // Process the tile
                    for (int d = 0; d < min(TILE_SIZE, kernel_size - tile_d); d++) {
                        for (int h = 0; h < min(TILE_SIZE, kernel_size - tile_h); h++) {
                            for (int w = 0; w < min(TILE_SIZE, kernel_size - tile_w); w++) {
                                sum += shared_input[d][h][w];
                            }
                        }
                    }
                    
                    __syncthreads();
                }
            }
        }

        int pool_size = kernel_size * kernel_size * kernel_size;
        output[index] = sum / static_cast<float>(pool_size);
        
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

    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;
    int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    avg_pool3d_shared_kernel<<<blocks, THREADS_PER_BLOCK>>>(
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
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) - shared memory version");
}