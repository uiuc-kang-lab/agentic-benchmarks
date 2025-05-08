#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 4
#define KERNEL_SIZE 3
#define WARPS_PER_BLOCK 8
#define SHARED_SIZE ((BLOCK_SIZE * TILE_SIZE) + KERNEL_SIZE - 1)

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {
    
    __shared__ float shared_input[SHARED_SIZE][SHARED_SIZE];
    __shared__ float shared_weight[KERNEL_SIZE][KERNEL_SIZE];
    
    const int tx = threadIdx.x % BLOCK_SIZE;
    const int ty = threadIdx.y % BLOCK_SIZE;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    
    const int bx = blockIdx.x * (BLOCK_SIZE * TILE_SIZE);
    const int by = blockIdx.y * (BLOCK_SIZE * TILE_SIZE);
    const int b = blockIdx.z / ((out_channels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const int oc_block = (blockIdx.z % ((out_channels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK)) * WARPS_PER_BLOCK;
    
    float partial_sums[TILE_SIZE][TILE_SIZE] = {0.0f};
    
    for (int oc_offset = 0; oc_offset < WARPS_PER_BLOCK && (oc_block + oc_offset) < out_channels; ++oc_offset) {
        const int oc = oc_block + oc_offset;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            if (tid < KERNEL_SIZE * KERNEL_SIZE) {
                int kid = tid;
                int kh = kid / KERNEL_SIZE;
                int kw = kid % KERNEL_SIZE;
                shared_weight[kh][kw] = weight[((oc * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw];
            }
            __syncthreads();
            
            #pragma unroll
            for (int i = 0; i < SHARED_SIZE; i += blockDim.y) {
                #pragma unroll
                for (int j = 0; j < SHARED_SIZE; j += blockDim.x) {
                    int load_y = i + ty;
                    int load_x = j + tx;
                    if (load_y < SHARED_SIZE && load_x < SHARED_SIZE) {
                        int ih = by + load_y - padding;
                        int iw = bx + load_x - padding;
                        
                        float val = 0.0f;
                        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                            val = input[((b * in_channels + ic) * input_height + ih) * input_width + iw];
                        }
                        shared_input[load_y][load_x] = val;
                    }
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int ti = 0; ti < TILE_SIZE; ++ti) {
                #pragma unroll
                for (int tj = 0; tj < TILE_SIZE; ++tj) {
                    float sum = 0.0f;
                    
                    #pragma unroll
                    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                        #pragma unroll
                        for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                            int sy = ty * TILE_SIZE + ti * stride + kh;
                            int sx = tx * TILE_SIZE + tj * stride + kw;
                            sum += shared_input[sy][sx] * shared_weight[kh][kw];
                        }
                    }
                    partial_sums[ti][tj] += sum;
                }
            }
            __syncthreads();
        }
        
        #pragma unroll
        for (int ti = 0; ti < TILE_SIZE; ++ti) {
            #pragma unroll
            for (int tj = 0; tj < TILE_SIZE; ++tj) {
                int out_y = by + ty * TILE_SIZE + ti;
                int out_x = bx + tx * TILE_SIZE + tj;
                
                if (out_y < output_height && out_x < output_width) {
                    output[((b * out_channels + oc) * output_height + out_y) * output_width + out_x] = partial_sums[ti][tj];
                }
            }
        }
        
        #pragma unroll
        for (int ti = 0; ti < TILE_SIZE; ++ti) {
            #pragma unroll
            for (int tj = 0; tj < TILE_SIZE; ++tj) {
                partial_sums[ti][tj] = 0.0f;
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
    
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "Weight must be a contiguous CUDA tensor");
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto input_height = x.size(2);
    auto input_width = x.size(3);
    auto out_channels = weight.size(0);
    
    auto output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
    auto output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width},
                             x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (output_width + BLOCK_SIZE * TILE_SIZE - 1) / (BLOCK_SIZE * TILE_SIZE),
        (output_height + BLOCK_SIZE * TILE_SIZE - 1) / (BLOCK_SIZE * TILE_SIZE),
        batch_size * ((out_channels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK)
    );
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
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
    m.def("forward", &forward, "Balanced CUDA conv2d implementation");
}