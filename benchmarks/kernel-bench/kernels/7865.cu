#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_kernel(
const float* __restrict__ input,
const float* __restrict__ weight,
float* __restrict__ output,
const int batch_size,
const int in_channels,
const int out_channels,
const int height,
const int width,
const int kernel_h,
const int kernel_w,
const int stride,
const int padding) {

    const int TILE_SIZE = 16;  // Tile size for shared memory
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;

    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    const int b = blockIdx.z;
    const int oc = blockIdx.y;
    const int h = blockIdx.x / ((out_w + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.x % ((out_w + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE + threadIdx.x;

    if (h >= out_h || w >= out_w) return;

    float sum = 0.0f;

    const int h_start = max(0, -h * stride + padding);
    const int h_end = min(kernel_h, height - h * stride + padding);
    const int w_start = max(0, -w * stride + padding);
    const int w_end = min(kernel_w, width - w * stride + padding);

    // Process input in tiles for each input channel
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory
        for (int tile_h = h_start; tile_h < h_end; tile_h += TILE_SIZE) {
            for (int tile_w = w_start; tile_w < w_end; tile_w += TILE_SIZE) {
                if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {
                    const int h_in = h * stride + tile_h + threadIdx.y - padding;
                    const int w_in = w * stride + tile_w + threadIdx.x - padding;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        shared_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 
                            input[((b * in_channels + ic) * height + h_in) * width + w_in];
                    } else {
                        shared_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
                    }
                }
                
                // Load weight tile into shared memory
                if (threadIdx.y < kernel_h && threadIdx.x < kernel_w) {
                    shared_weight[threadIdx.y * kernel_w + threadIdx.x] = 
                        weight[((oc * in_channels + ic) * kernel_h + threadIdx.y) * kernel_w + threadIdx.x];
                }
                
                __syncthreads();

                // Compute convolution for this tile
                for (int kh = 0; kh < min(TILE_SIZE, h_end - tile_h); ++kh) {
                    for (int kw = 0; kw < min(TILE_SIZE, w_end - tile_w); ++kw) {
                        sum += shared_input[(threadIdx.y + kh) * TILE_SIZE + (threadIdx.x + kw)] *
                               shared_weight[kh * kernel_w + kw];
                    }
                }
                
                __syncthreads();
            }
        }
    }

    if (h < out_h && w < out_w) {
        output[((b * out_channels + oc) * out_h + h) * out_w + w] = sum;
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
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias, {stride, stride},
                           {padding, padding}, {dilation, dilation}, groups);
    }
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_h = (height + 2 * padding - kernel_h) / stride + 1;
    auto out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w},
                              x.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_h * out_w + threads - 1) / threads;
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
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
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}
