#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {

    __shared__ float shared_weight[32][32];  // Cache for weights

    // Calculate output position
    const int batch_idx = blockIdx.z;
    const int out_ch = blockIdx.y;
    // Define tile dimensions: each block handles a tile of output rows
    const int tile_rows = BLOCK_SIZE / out_width;  // assume BLOCK_SIZE is a multiple of out_width
    const int tile_row = threadIdx.x / out_width;
    const int tile_col = threadIdx.x % out_width;
    const int h_out = blockIdx.x * tile_rows + tile_row;
    const int w_out = tile_col;

    // Pre-load bias
    float bias_val = bias ? bias[out_ch] : 0.0f;

    // Process multiple rows per thread block
    for (int h_out_offset = 0; h_out_offset < BLOCK_SIZE / out_width && h_out_start + h_out_offset < out_height; h_out_offset++) {
        const int h_out = h_out_start + h_out_offset;
        
        float sum = bias_val;
        
        // Process input channels in chunks to fit in shared memory
        for (int c_in_block = 0; c_in_block < in_channels; c_in_block += 32) {
            
            // Load weights into shared memory
            if (tid < kernel_height * kernel_width && c_in_block + warp_id < in_channels) {
                shared_weight[warp_id][tid] = weight[
                    ((out_ch * in_channels + c_in_block + warp_id) * kernel_height * kernel_width) + tid];
            }
            __syncthreads();

            // Compute convolution with cached weights
            for (int c_in_offset = 0; c_in_offset < min(32, in_channels - c_in_block); c_in_offset++) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    const int h_in = h_out * stride - padding + kh;
                    if (h_in >= 0 && h_in < in_height) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            for (int w_out = w_out_base; w_out < out_width; w_out += WARP_SIZE) {
                                const int w_in = w_out * stride - padding + kw;
                                if (w_in >= 0 && w_in < in_width) {
                                    const int in_idx = ((batch_idx * in_channels + (c_in_block + c_in_offset)) * 
                                                      in_height + h_in) * in_width + w_in;
                                    sum += input[in_idx] * shared_weight[c_in_offset][kh * kernel_width + kw];
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Write output with coalesced access pattern
        for (int w_out = w_out_base; w_out < out_width; w_out += WARP_SIZE) {
            const int out_idx = ((batch_idx * out_channels + out_ch) * out_height + h_out) * 
                               out_width + w_out;
            output[out_idx] = sum;
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
    
    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias.has_value() ? bias.value() : torch::Tensor(),
                           {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    const auto out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width},
                              x.options());

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(
        (out_height + (BLOCK_SIZE/out_width) - 1) / (BLOCK_SIZE/out_width),
        out_channels,
        batch_size
    );
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        out_height, out_width,
        stride, padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}