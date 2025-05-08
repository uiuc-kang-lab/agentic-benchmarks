#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define TILE_WIDTH 4
#define SMEM_STRIDE 32

__global__ void conv2d_warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {

    __shared__ float smem_input[SMEM_STRIDE * SMEM_STRIDE];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int batch_idx = blockIdx.z;
    const int out_ch = blockIdx.y;
    
    const int tile_x = (blockIdx.x * TILE_WIDTH * (BLOCK_SIZE/WARP_SIZE) + (warp_id * TILE_WIDTH)) * WARP_SIZE + lane_id;
    const int tile_y = blockIdx.x / ((output_width + TILE_WIDTH*WARP_SIZE - 1)/(TILE_WIDTH*WARP_SIZE));

    float partial_sums[TILE_WIDTH][TILE_WIDTH] = {0.0f};

    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int i = threadIdx.x; i < SMEM_STRIDE * SMEM_STRIDE; i += BLOCK_SIZE) {
            int smem_y = i / SMEM_STRIDE;
            int smem_x = i % SMEM_STRIDE;
            int in_y = tile_y + smem_y - padding;
            int in_x = tile_x + smem_x - padding;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                val = input[((batch_idx * in_channels + in_ch) * input_height + in_y) * input_width + in_x];
            }
            smem_input[smem_y * SMEM_STRIDE + smem_x] = val;
        }
        __syncthreads();

        #pragma unroll
        for (int ty = 0; ty < TILE_WIDTH; ty++) {
            #pragma unroll
            for (int tx = 0; tx < TILE_WIDTH; tx++) {
                float sum = 0.0f;
                
                #pragma unroll
                for (int ky = 0; ky < kernel_size; ky++) {
                    #pragma unroll
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int smem_y = ty * stride + ky;
                        int smem_x = tx * stride + kx;
                        float in_val = smem_input[smem_y * SMEM_STRIDE + smem_x];
                        float w_val = weight[((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx];
                        sum += in_val * w_val;
                    }
                }
                partial_sums[ty][tx] += sum;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ty = 0; ty < TILE_WIDTH; ty++) {
        #pragma unroll
        for (int tx = 0; tx < TILE_WIDTH; tx++) {
            float sum = partial_sums[ty][tx];
            
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (lane_id == 0) {
                int out_y = tile_y + ty;
                int out_x = tile_x + tx * WARP_SIZE;
                if (out_y < output_height && out_x < output_width) {
                    int out_idx = ((batch_idx * out_channels + out_ch) * output_height + out_y) * output_width + out_x;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, x.options());

    const int grid_x = (output_width + TILE_WIDTH * WARP_SIZE - 1) / (TILE_WIDTH * WARP_SIZE);
    const int grid_y = out_channels;
    const int grid_z = batch_size;
    
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(BLOCK_SIZE);

    conv2d_warp_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width,
        stride,
        padding);
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward convolution with warp-level optimizations");
}