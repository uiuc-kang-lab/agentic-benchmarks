#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define BLOCK_SIZE 16

__global__ void optimized_conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int batch_offset) {

    extern __shared__ float shared_mem[];
    float* input_tile = shared_mem;
    float* weight_tile = &shared_mem[TILE_SIZE * TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + tx;
    const int h_out = blockIdx.y * blockDim.y + ty;
    const int oc = blockIdx.z;
    
    if (w_out >= width_out || h_out >= height_out) return;

    const float bias_val = bias ? bias[oc] : 0.0f;
    float sum = bias_val;

    const int h_in_start = h_out * stride - pad_h;
    const int w_in_start = w_out * stride - pad_w;
    const int b = batch_offset;
    const int batch_stride = in_channels * input_height * input_width;
    const int out_batch_stride = out_channels * height_out * width_out;

    #pragma unroll 2
    for (int ic = 0; ic < in_channels; ic += BLOCK_SIZE) {
        __syncthreads();
        
        // Load input tile
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        for (int j = 0; j < TILE_SIZE; j += blockDim.x) {
            int h_in = h_in_start + i + ty;
            int w_in = w_in_start + j + tx;
            if (i + ty < TILE_SIZE && j + tx < TILE_SIZE) {
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    input_tile[(i + ty) * TILE_SIZE + (j + tx)] = 
                        x[b * batch_stride + ic * input_height * input_width + 
                          h_in * input_width + w_in];
                } else {
                    input_tile[(i + ty) * TILE_SIZE + (j + tx)] = 0.0f;
                }
            }
        }
    }

    // Load weight tile
    for (int i = 0; i < kernel_h; i += blockDim.y) {
        for (int j = 0; j < kernel_w; j += blockDim.x) {
            if (ty + i < kernel_h && tx + j < kernel_w) {
                weight_tile[(i + ty) * kernel_w + (j + tx)] = 
                    weight[oc * in_channels * kernel_h * kernel_w +
                          ic * kernel_h * kernel_w + 
                          (i + ty) * kernel_w + (j + tx)];
            }
        }
    }
    __syncthreads();

    // Compute convolution
    for (int k = 0; k < min(BLOCK_SIZE, in_channels - ic); ++k) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            #pragma unroll 4
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int h_in_idx = kh * dilation_h;
                const int w_in_idx = kw * dilation_w;
                
                if (h_in_start + h_in_idx >= 0 && 
                    h_in_start + h_in_idx < input_height && 
                    w_in_start + w_in_idx >= 0 && 
                    w_in_start + w_in_idx < input_width) {
                    sum += input_tile[(h_in_idx) * TILE_SIZE + (w_in_idx)] * 
                           weight_tile[kh * kernel_w + kw];
                }
            }
        }
    }
    }

    if (w_out < width_out && h_out < height_out) {
        output[b * out_batch_stride + oc * height_out * width_out + h_out * width_out + w_out] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight must be a contiguous CUDA tensor");

    auto dims = x.sizes();
    int batch_size = dims[0];
    int in_channels = dims[1];
    int input_height = dims[2];
    int input_width = dims[3];
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    const int num_streams = std::min(4, batch_size);
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threads(16, 16);
    dim3 blocks((width_out + threads.x - 1) / threads.x,
                (height_out + threads.y - 1) / threads.y,
                out_channels);

    const int shared_mem_size = (TILE_SIZE * TILE_SIZE + kernel_h * kernel_w) * sizeof(float);

    for (int b = 0; b < batch_size; b++) {
        optimized_conv2d_kernel<<<blocks, threads, shared_mem_size, streams[b % num_streams]>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(),
            bias.has_value() ? bias->data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, input_height, input_width,
            out_channels, kernel_h, kernel_w, height_out, width_out,
            stride, pad_h, pad_w, dilation_h, dilation_w, b);
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}