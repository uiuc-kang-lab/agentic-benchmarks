#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define TILE_WIDTH 32
#define BLOCK_HEIGHT 8

__global__ void conv2d_coalesced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    __shared__ float shared_input[BLOCK_HEIGHT][(TILE_WIDTH + 2 * padding)];
    __shared__ float shared_weight[BLOCK_HEIGHT][TILE_WIDTH];

    // Calculate output position
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_WIDTH;
    const int by = blockIdx.y * BLOCK_HEIGHT;
    const int batch_idx = blockIdx.z;

    // Calculate output coordinates - ensures coalesced access pattern
    const int out_x = bx + tx;
    const int out_y = by + ty;
    
    float sum = 0.0f;

    // Process input channels in tiles
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory with padding
        for (int i = ty; i < BLOCK_HEIGHT; i += blockDim.y) {
            for (int j = tx; j < TILE_WIDTH; j += blockDim.x) {
                const int in_y = (by + i) * stride - padding;
                const int in_x = (bx + j) * stride - padding;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    shared_input[i][j] = input[
                        ((batch_idx * in_channels + ic) * in_height + in_y) * in_width + in_x
                    ];
                } else {
                    shared_input[i][j] = 0.0f;
                }
            }
        }

        // Load weight tile into shared memory
        for (int oc = 0; oc < out_channels; ++oc) {
            if (ty < kernel_size && tx < kernel_size) {
                shared_weight[ty][tx] = weight[
                    ((oc * in_channels + ic) * kernel_size + ty) * kernel_size + tx
                ];
            }
        }
        
        __syncthreads();

        // Compute convolution for this tile
        if (out_x < out_width && out_y < out_height) {
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int in_y = out_y * stride + ky * dilation - padding;
                    const int in_x = out_x * stride + kx * dilation - padding;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        sum += shared_input[ky][kx] * shared_weight[ky][kx];
                    }
                }
            }
        }
        
        __syncthreads();
    }

    // Write output with coalesced access pattern
    if (out_x < out_width && out_y < out_height) {
        const int out_idx = (
            (batch_idx * out_channels + blockIdx.y) * out_height + out_y
        ) * out_width + out_x;
        
        if (bias != nullptr) {
            sum += bias[blockIdx.y];
        }
        
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Configure kernel launch parameters for coalesced access
    dim3 threads(WARP_SIZE, BLOCK_HEIGHT);
    dim3 blocks(
        (out_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        batch_size
    );

    conv2d_coalesced_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding, dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with coalesced memory access");
}