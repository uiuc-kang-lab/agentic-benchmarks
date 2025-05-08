#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define TILE_SIZE 32

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w) {
    
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int out_ch = blockIdx.z % out_channels;
    const int batch = blockIdx.z / out_channels;
    
    if (batch >= batch_size) return;
    
    float local_sum = 0.0f;
    
    // Process input in tiles
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_height; kh += TILE_SIZE) {
            for (int kw = 0; kw < kernel_width; kw += TILE_SIZE) {
                // Load weight tile into shared memory
                if (tx < TILE_SIZE && ty < TILE_SIZE) {
                    if ((kh + ty) < kernel_height && (kw + tx) < kernel_width) {
                        shared_weight[ty][tx] = weight[
                            in_ch * out_channels * kernel_height * kernel_width +
                            out_ch * kernel_height * kernel_width +
                            (kh + ty) * kernel_width + (kw + tx)];
                    } else {
                        shared_weight[ty][tx] = 0.0f;
                    }
                }
                __syncthreads();
                
                const int out_x = bx + tx;
                const int out_y = by + ty;
                
                if (out_x < output_width && out_y < output_height) {
                    for (int i = 0; i < min(TILE_SIZE, kernel_height - kh); ++i) {
                        for (int j = 0; j < min(TILE_SIZE, kernel_width - kw); ++j) {
                            int in_x = (out_x + padding_w - (kw + j)) / stride_w;
                            int in_y = (out_y + padding_h - (kh + i)) / stride_h;
                            
                            if (in_x >= 0 && in_x < input_width &&
                                in_y >= 0 && in_y < input_height &&
                                (out_x + padding_w - (kw + j)) % stride_w == 0 &&
                                (out_y + padding_h - (kh + i)) % stride_h == 0) {
                                
                                float input_val = input[
                                    batch * in_channels * input_height * input_width +
                                    in_ch * input_height * input_width +
                                    in_y * input_width + in_x];
                                
                                local_sum += input_val * shared_weight[i][j];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    
    const int out_x = bx + tx;
    const int out_y = by + ty;
    
    if (out_x < output_width && out_y < output_height) {
        const int out_idx = batch * out_channels * output_height * output_width +
                            out_ch * output_height * output_width +
                            out_y * output_width + out_x;
        output[out_idx] = local_sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);
    
    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] +
                              kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] +
                             kernel_width + output_padding[1];
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                              x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}