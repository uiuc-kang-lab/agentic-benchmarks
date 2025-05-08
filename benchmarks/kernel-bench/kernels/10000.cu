#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 16

template<int KERNEL_SIZE>
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    extern __shared__ float shared_input[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int b = bz / out_channels;
    const int c = bz % out_channels;
    const int g = c / channels_per_group;
    const int m = c % channels_per_group;
    
    // Calculate output coordinates
    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = by * TILE_SIZE + ty;
    
    // Calculate input tile dimensions including halo region
    const int input_tile_width = TILE_SIZE * stride_w + (KERNEL_SIZE - 1) * dilation_w;
    const int input_tile_height = TILE_SIZE * stride_h + (KERNEL_SIZE - 1) * dilation_h;
    
    // Load input tile into shared memory
    for (int i = ty; i < input_tile_height; i += BLOCK_SIZE) {
        for (int j = tx; j < input_tile_width; j += BLOCK_SIZE) {
            const int in_y = by * TILE_SIZE * stride_h + i - padding_h;
            const int in_x = bx * TILE_SIZE * stride_w + j - padding_w;
            
            float value = 0.0f;
            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w && b < batch_size) {
                value = input[((b * in_channels + g) * in_h + in_y) * in_w + in_x];
            }
            shared_input[i * input_tile_width + j] = value;
        }
    }
    __syncthreads();
    
    if (out_x < out_w && out_y < out_h && b < batch_size) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int sh_y = ty * stride_h + kh * dilation_h;
                const int sh_x = tx * stride_w + kw * dilation_w;
                
                const float in_val = shared_input[sh_y * input_tile_width + sh_x];
                const float w_val = weight[((g * channels_per_group + m) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw];
                sum += in_val * w_val;
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[((b * out_channels + c) * out_h + out_y) * out_w + out_x] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_size = weight.size(2);
    
    TORCH_CHECK(kernel_size == weight.size(3), "Kernel must be square");
    
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_size - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_size - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    // Calculate shared memory size
    int input_tile_width = TILE_SIZE * stride_w + (kernel_size - 1) * dilation_w;
    int input_tile_height = TILE_SIZE * stride_h + (kernel_size - 1) * dilation_h;
    int shared_mem_size = input_tile_width * input_tile_height * sizeof(float);

    switch(kernel_size) {
        case 3:
            depthwise_conv2d_kernel<3><<<blocks, threads, shared_mem_size>>>(
                x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
                output.data_ptr<float>(), batch_size, in_channels, in_h, in_w,
                out_channels, out_h, out_w, stride_h, stride_w, padding_h,
                padding_w, dilation_h, dilation_w, groups, channels_per_group);
            break;
        case 5:
            depthwise_conv2d_kernel<5><<<blocks, threads, shared_mem_size>>>(
                x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
                output.data_ptr<float>(), batch_size, in_channels, in_h, in_w,
                out_channels, out_h, out_w, stride_h, stride_w, padding_h,
                padding_w, dilation_h, dilation_w, groups, channels_per_group);
            break;
        default:
            TORCH_CHECK(false, "Unsupported kernel size");
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward (CUDA)");
}