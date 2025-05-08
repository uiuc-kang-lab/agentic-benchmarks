#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Using dynamic shared memory to accommodate varying kernel sizes
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
    const int kernel_size,
    const int stride,
    const int padding) {
    
    // dynamic shared memory allocation
    extern __shared__ scalar_t shared_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;
    
    int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    int w_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    // s_size: shared memory tile dimension along one axis
    int s_size = TILE_SIZE + kernel_size - 1;
    // Determine the top-left corner of the input region for this block
    int row_start = blockIdx.y * TILE_SIZE * stride - padding;
    int col_start = blockIdx.x * TILE_SIZE * stride - padding;

    for (int ic = 0; ic < in_channels; ic++) {
        // Load the input tile (including halo) into shared memory in a tiled manner
        for (int i = ty; i < s_size; i += TILE_SIZE) {
            for (int j = tx; j < s_size; j += TILE_SIZE) {
                int in_r = row_start + i;
                int in_c = col_start + j;
                if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                    int in_idx = ((b * in_channels + ic) * height + in_r) * width + in_c;
                    shared_input[i * s_size + j] = input[in_idx];
                } else {
                    shared_input[i * s_size + j] = 0;
                }
            }
        }
        __syncthreads();
        
        // Each thread computes one output element if within bounds
        int out_row = blockIdx.y * TILE_SIZE + ty;
        int out_col = blockIdx.x * TILE_SIZE + tx;
        if (out_row < h_out && out_col < w_out) {
            scalar_t sum = 0;
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    // Access the corresponding element from shared memory
                    sum += shared_input[(ty + kh) * s_size + (tx + kw)] * weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                }
            }
            int out_idx = ((b * out_channels + oc) * h_out + out_row) * w_out + out_col;
            output[out_idx] += sum;
        }
        __syncthreads();
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
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto h_out = (height + 2 * padding - kernel_size) / stride + 1;
    auto w_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((w_out + TILE_SIZE - 1) / TILE_SIZE,
                (h_out + TILE_SIZE - 1) / TILE_SIZE,
                batch_size * out_channels);
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv2d_forward_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding);
    }));
    
    if (bias.has_value()) {
        output += bias.value().view({1, out_channels, 1, 1});
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with optional bias");
}