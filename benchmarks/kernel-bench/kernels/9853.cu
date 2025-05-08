#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define SMALL_KERNEL_THRESHOLD 64 // Threshold for input size to switch between implementations

__global__ void depthwise_conv2d_tiled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weight = shared_mem + (TILE_WIDTH + kernel_size - 1) * (TILE_HEIGHT + kernel_size - 1);

    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Load weights into shared memory
    if (threadIdx.y == 0 && threadIdx.x < kernel_size * kernel_size) {
        s_weight[threadIdx.x] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) + 
            threadIdx.x
        ];
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_WIDTH + tx;
    int out_y = blockIdx.y * TILE_HEIGHT + ty;
    
    if (out_x >= output_w || out_y >= output_h) return;

    // Compute input positions
    int in_x = out_x * stride - padding;
    int in_y = out_y * stride - padding;

    float sum = 0.0f;
    
    if (kernel_size == 3) {
        // Optimized 3x3 kernel path
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            int in_y_k = in_y + ky;
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                int in_x_k = in_x + kx;
                if (in_y_k >= 0 && in_y_k < input_h && in_x_k >= 0 && in_x_k < input_w) {
                    float in_val = input[b * (in_channels * input_h * input_w) +
                                       in_ch * (input_h * input_w) +
                                       in_y_k * input_w + in_x_k];
                    sum += in_val * s_weight[ky * 3 + kx];
                }
            }
        }
    } else {
        // Generic kernel size path
        for (int ky = 0; ky < kernel_size; ++ky) {
            int in_y_k = in_y + ky;
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x_k = in_x + kx;
                if (in_y_k >= 0 && in_y_k < input_h && in_x_k >= 0 && in_x_k < input_w) {
                    float in_val = input[b * (in_channels * input_h * input_w) +
                                       in_ch * (input_h * input_w) +
                                       in_y_k * input_w + in_x_k];
                    sum += in_val * s_weight[ky * kernel_size + kx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[b * (out_channels * output_h * output_w) +
           oc * (output_h * output_w) +
           out_y * output_w +
           out_x] = sum;
}

__global__ void depthwise_conv2d_simple(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_h * output_w) return;

    int w_out = idx % output_w;
    int h_out = (idx / output_w) % output_h;
    int oc = (idx / (output_w * output_h)) % out_channels;
    int b = idx / (output_w * output_h * out_channels);

    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;
            
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                sum += input[b * (in_channels * input_h * input_w) +
                           in_ch * (input_h * input_w) +
                           h_in * input_w + w_in] *
                      weight[in_ch * (channels_per_group * kernel_size * kernel_size) +
                            weight_ch * (kernel_size * kernel_size) +
                            kh * kernel_size + kw];
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[idx] = sum;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;
    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    if (input_h * input_w <= SMALL_KERNEL_THRESHOLD) {
        int total_elements = batch_size * out_channels * output_h * output_w;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        depthwise_conv2d_simple<<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), batch_size, in_channels, input_h, input_w,
            out_channels, output_h, output_w, kernel_size, stride, padding,
            channels_per_group
        );
    } else {
        dim3 block(16, 16);
        dim3 grid((output_w + TILE_WIDTH - 1) / TILE_WIDTH,
                 (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 batch_size * out_channels);
                 
        size_t shared_mem_size = ((TILE_WIDTH + kernel_size - 1) * 
                                 (TILE_HEIGHT + kernel_size - 1) + 
                                 kernel_size * kernel_size) * sizeof(float);

        depthwise_conv2d_tiled<<<grid, block, shared_mem_size>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), batch_size, in_channels, input_h, input_w,
            out_channels, output_h, output_w, kernel_size, stride, padding,
            channels_per_group
        );
    }

    return output;
}