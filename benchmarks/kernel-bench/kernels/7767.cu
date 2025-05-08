#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void conv2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding) {
    
    extern __shared__ float shared_input[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = by * TILE_SIZE + ty;
    const int batch_idx = bz / out_channels;
    const int out_ch = bz % out_channels;
    
    if (out_x >= width || out_y >= height) return;
    
    scalar_t sum = 0;
    
    const int input_start_x = out_x * stride - padding;
    const int input_start_y = out_y * stride - padding;
    
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        // Load input tile into shared memory
        const int tile_h = TILE_SIZE + kernel_h - 1;
        const int tile_w = TILE_SIZE + kernel_w - 1;
        
        for (int i = ty; i < tile_h; i += TILE_SIZE) {
            for (int j = tx; j < tile_w; j += TILE_SIZE) {
                int y = input_start_y + i;
                int x = input_start_x + j;
                
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    shared_input[i * tile_w + j] = input[
                        ((batch_idx * in_channels + in_ch) * height + y) * width + x
                    ];
                } else {
                    shared_input[i * tile_w + j] = 0;
                }
            }
        }
        __syncthreads();
        
        // Compute convolution
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int y = ty * stride + ky;
                int x = tx * stride + kx;
                
                sum += shared_input[y * tile_w + x] *
                       weight[((out_ch * in_channels + in_ch) * kernel_h + ky) * kernel_w + kx];
            }
        }
        __syncthreads();
    }
    
    output[((batch_idx * out_channels + out_ch) * height + out_y) * width + out_x] = sum;
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
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    
    const auto output_height = (height + 2 * padding - kernel_h) / stride + 1;
    const auto output_width = (width + 2 * padding - kernel_w) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                              x.options());
    
    const int threads = TILE_SIZE;
    const dim3 blocks(
        (output_width + threads - 1) / threads,
        (output_height + threads - 1) / threads,
        batch_size * out_channels
    );
    const dim3 threads_per_block(threads, threads);
    
    const int shared_mem_size = ((TILE_SIZE + kernel_h - 1) * (TILE_SIZE + kernel_w - 1)) * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv2d_forward_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride,
            padding
        );
    }));
    
    if (bias.has_value()) {
        output += bias.value().view({1, out_channels, 1, 1});
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory");
}