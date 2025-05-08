#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 16

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
    int kernel_h,
    int kernel_w,
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
    
    // Calculate input tile dimensions including halo region
    const int tile_w = TILE_SIZE + (kernel_w - 1) * dilation_w;
    const int tile_h = TILE_SIZE + (kernel_h - 1) * dilation_h;
    
    // Calculate base input position
    const int in_x = bx * TILE_SIZE * stride_w - padding_w;
    const int in_y = by * TILE_SIZE * stride_h - padding_h;
    
    // Load input tile into shared memory
    #pragma unroll
    for (int i = ty; i < tile_h; i += BLOCK_SIZE) {
        #pragma unroll
        for (int j = tx; j < tile_w; j += BLOCK_SIZE) {
            const int y = in_y + i;
            const int x = in_x + j;
            
            float val = 0.0f;
            if (y >= 0 && y < in_h && x >= 0 && x < in_w) {
                val = input[((b * in_channels + g) * in_h + y) * in_w + x];
            }
            shared_input[i * tile_w + j] = val;
        }
    }
    
    // Single synchronization point after loading shared memory
    __syncthreads();
    
    // Compute output
    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = by * TILE_SIZE + ty;
    
    if (out_x < out_w && out_y < out_h) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++) {
            #pragma unroll
            for (int kw = 0; kw < kernel_w; kw++) {
                const int sh_y = ty * stride_h + kh * dilation_h;
                const int sh_x = tx * stride_w + kw * dilation_w;
                
                const float in_val = shared_input[sh_y * tile_w + sh_x];
                const float w_val = weight[((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw];
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

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    // Calculate shared memory size
    int tile_w = TILE_SIZE + (kernel_w - 1) * dilation_w;
    int tile_h = TILE_SIZE + (kernel_h - 1) * dilation_h;
    int shared_mem_size = tile_w * tile_h * sizeof(float);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    depthwise_conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward (CUDA)");
}