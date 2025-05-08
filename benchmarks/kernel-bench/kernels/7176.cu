#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16
#define BLOCK_SIZE 16

__global__ void conv2d_tile_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    
    // Calculate output position
    const int batch_idx = blockIdx.x;
    const int out_ch_idx = blockIdx.y;
    const int tile_row = blockIdx.z / ((out_width + TILE_SIZE - 1) / TILE_SIZE);
    const int tile_col = blockIdx.z % ((out_width + TILE_SIZE - 1) / TILE_SIZE);
    const int out_row = tile_row * TILE_SIZE + threadIdx.y;
    const int out_col = tile_col * TILE_SIZE + threadIdx.x;
    
    // Early exit if outside output dimensions
    if (out_row >= out_height || out_col >= out_width) return;
    
    float sum = 0.0f;
    
    // Process input channels
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        // Load input tile into shared memory
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                const int in_row = out_row * stride - padding + ky * dilation;
                const int in_col = out_col * stride - padding + kx * dilation;
                
                if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
                    const int in_idx = ((batch_idx * in_channels + in_ch) * in_height + in_row) * in_width + in_col;
                    shared_input[threadIdx.y][threadIdx.x] = input[in_idx];
                } else {
                    shared_input[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                __syncthreads();
                
                // Compute partial sum for current kernel position
                const int weight_idx = ((out_ch_idx * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                sum += shared_input[threadIdx.y][threadIdx.x] * weight[weight_idx];
                
                __syncthreads();
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        sum += bias[out_ch_idx];
    }
    
    // Write output
    if (out_row < out_height && out_col < out_width) {
        const int out_idx = ((batch_idx * out_channels + out_ch_idx) * out_height + out_row) * out_width + out_col;
        output[out_idx] = sum;
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
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    
    TORCH_CHECK(groups == 1, "Only groups==1 is supported");
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    // Calculate grid dimensions
    const int grid_height = (out_height + TILE_SIZE - 1) / TILE_SIZE;
    const int grid_width = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        batch_size,
        out_channels,
        grid_height * grid_width
    );
    
    conv2d_tile_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled CUDA convolution with optimized thread indexing");
}