#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define KERNEL_SIZE 3
#define TILE_SIZE 8
#define SHARED_SIZE (TILE_SIZE + KERNEL_SIZE - 1)

__global__ void conv2d_coalesced_kernel(
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
    
    // Calculate global position
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int b = blockIdx.z / out_channels;
    const int oc = blockIdx.z % out_channels;
    
    // Coalesced global position
    const int x = bx + tx;
    const int y = by + ty;
    
    float sum = 0.0f;
    
    // Load weights into shared memory (coalesced access pattern)
    if (threadIdx.x < KERNEL_SIZE * KERNEL_SIZE) {
        int kid = threadIdx.x;
        int krow = kid / KERNEL_SIZE;
        int kcol = kid % KERNEL_SIZE;
        shared_weight[krow][kcol] = weight[oc * in_channels * KERNEL_SIZE * KERNEL_SIZE + kid];
    }
    __syncthreads();
    
    // Process input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory with coalesced access
        for (int i = threadIdx.x; i < SHARED_SIZE * SHARED_SIZE; i += BLOCK_SIZE) {
            int row = i / SHARED_SIZE;
            int col = i % SHARED_SIZE;
            int ih = by + row - padding;
            int iw = bx + col - padding;
            
            float val = 0.0f;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                // Coalesced read from input
                val = input[((b * in_channels + ic) * input_height + ih) * input_width + iw];
            }
            shared_input[row][col] = val;
        }
        __syncthreads();
        
        if (x < output_width && y < output_height) {
            // Compute convolution with coalesced memory access
            #pragma unroll
            for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
                #pragma unroll
                for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                    sum += shared_input[ty * stride + ki][tx * stride + kj] * 
                           shared_weight[ki][kj];
                }
            }
        }
        __syncthreads();
    }
    
    // Coalesced write to output
    if (x < output_width && y < output_height) {
        output[((b * out_channels + oc) * output_height + y) * output_width + x] = sum;
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
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto input_height = x.size(2);
    auto input_width = x.size(3);
    auto out_channels = weight.size(0);
    
    auto output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
    auto output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width},
                             x.options());
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(
        (output_width + TILE_SIZE - 1) / TILE_SIZE,
        (output_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_coalesced_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Coalesced CUDA conv2d implementation");
}