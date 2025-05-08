#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define SMALL_TILE_SIZE 16
#define LARGE_TILE_SIZE 32

template<int TILE_H, int TILE_W>
__global__ void conv2d_adaptive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
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

    extern __shared__ float shared_input[];
    
    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    
    const int num_tiles_x = (out_width + TILE_W - 1) / TILE_W;
    const int tile_idx = blockIdx.z;
    const int tile_row = tile_idx / num_tiles_x;
    const int tile_col = tile_idx % num_tiles_x;

    const int out_start_y = tile_row * TILE_H;
    const int out_start_x = tile_col * TILE_W;
    const int out_y = out_start_y + threadIdx.y;
    const int out_x = out_start_x + threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    const int sh_height = (TILE_H - 1) * stride + (kernel_size - 1) * dilation + 1;
    const int sh_width = (TILE_W - 1) * stride + (kernel_size - 1) * dilation + 1;
    const int in_tile_y = out_start_y * stride - padding;
    const int in_tile_x = out_start_x * stride - padding;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
        const int total_sh_elems = sh_height * sh_width;
        
        for (int idx = thread_idx; idx < total_sh_elems; idx += blockDim.x * blockDim.y) {
            const int sh_y = idx / sh_width;
            const int sh_x = idx % sh_width;
            const int in_y = in_tile_y + sh_y;
            const int in_x = in_tile_x + sh_x;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                const int input_idx = ((batch_idx * in_channels + ic) * in_height + in_y) * in_width + in_x;
                val = input[input_idx];
            }
            shared_input[sh_y * sh_width + sh_x] = val;
        }
        __syncthreads();

        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            const int sh_y = threadIdx.y * stride + ky * dilation;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                const int sh_x = threadIdx.x * stride + kx * dilation;
                const float in_val = shared_input[sh_y * sh_width + sh_x];
                const int weight_idx = ((out_channel * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                sum += in_val * weight[weight_idx];
            }
        }
        __syncthreads();
    }

    if (bias) {
        sum += bias[out_channel];
    }
    
    const int out_idx = ((batch_idx * out_channels + out_channel) * out_height + out_y) * out_width + out_x;
    output[out_idx] = sum;
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

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    if (in_height <= 32 && in_width <= 32) {
        const int num_tiles_x = (out_width + SMALL_TILE_SIZE - 1) / SMALL_TILE_SIZE;
        const int num_tiles_y = (out_height + SMALL_TILE_SIZE - 1) / SMALL_TILE_SIZE;
        
        dim3 block(SMALL_TILE_SIZE, SMALL_TILE_SIZE);
        dim3 grid(batch, out_channels, num_tiles_x * num_tiles_y);
        
        const int sh_height = (SMALL_TILE_SIZE - 1) * stride + (kernel_size - 1) * dilation + 1;
        const int sh_width = (SMALL_TILE_SIZE - 1) * stride + (kernel_size - 1) * dilation + 1;
        const size_t shared_mem_size = sh_height * sh_width * sizeof(float);

        conv2d_adaptive_kernel<SMALL_TILE_SIZE, SMALL_TILE_SIZE><<<grid, block, shared_mem_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            kernel_size, stride, padding, dilation);
    } else {
        const int num_tiles_x = (out_width + LARGE_TILE_SIZE - 1) / LARGE_TILE_SIZE;
        const int num_tiles_y = (out_height + LARGE_TILE_SIZE - 1) / LARGE_TILE_SIZE;
        
        dim3 block(LARGE_TILE_SIZE, LARGE_TILE_SIZE);
        dim3 grid(batch, out_channels, num_tiles_x * num_tiles_y);
        
        const int sh_height = (LARGE_TILE_SIZE - 1) * stride + (kernel_size - 1) * dilation + 1;
        const int sh_width = (LARGE_TILE_SIZE - 1) * stride + (kernel_size - 1) * dilation + 1;
        const size_t shared_mem_size = sh_height * sh_width * sizeof(float);

        conv2d_adaptive_kernel<LARGE_TILE_SIZE, LARGE_TILE_SIZE><<<grid, block, shared_mem_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            kernel_size, stride, padding, dilation);
    }

    return output;
}