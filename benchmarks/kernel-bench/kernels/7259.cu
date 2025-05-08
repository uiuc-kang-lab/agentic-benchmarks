#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define KERNEL_SIZE 3
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __inline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void conv2d_kernel(
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
    const int stride,
    const int padding) {
    
    __shared__ float smem_input[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    __shared__ float smem_weights[KERNEL_SIZE][KERNEL_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int oc = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        // Load weights for this output channel/input channel
        if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
            smem_weights[tx][ty] = weight[((oc * in_channels + ic) * KERNEL_SIZE + tx) * KERNEL_SIZE + ty];
        }
        __syncthreads();

        // Load input tile with halo
        for (int i = ty; i < TILE_SIZE + KERNEL_SIZE - 1; i += blockDim.y) {
            for (int j = tx; j < TILE_SIZE + KERNEL_SIZE - 1; j += blockDim.x) {
                int h = by - padding + i;
                int w = bx - padding + j;
                if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
                    smem_input[i][j] = input[((b * in_channels + ic) * in_height + h) * in_width + w];
                } else {
                    smem_input[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute convolution
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                sum += smem_input[ty + kh][tx + kw] * smem_weights[kh][kw];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction
    sum = warp_reduce(sum);

    // Write output
    if (tx == 0 && ty == 0) {
        const int out_h = by + (ty % TILE_SIZE);
        const int out_w = bx + (tx % TILE_SIZE);
        if (out_h < out_height && out_w < out_width) {
            output[((b * out_channels + oc) * out_height + out_h) * out_width + out_w] = sum;
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
    const int out_height = (in_height + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        stride,
        padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with shared memory optimizations");
}