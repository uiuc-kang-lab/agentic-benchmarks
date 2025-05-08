#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Calculate output position
    const int out_idx = bid;
    if (out_idx >= batch_size * out_channels * out_h * out_w) return;
    
    const int w_out = out_idx % out_w;
    const int h_out = (out_idx / out_w) % out_h;
    const int c_out = (out_idx / (out_h * out_w)) % out_channels;
    const int b = out_idx / (out_channels * out_h * out_w);
    
    const int g = c_out / channels_per_group;
    const int m = c_out % channels_per_group;

    // Compute partial sum in registers
    float reg_sum = 0.0f;
    const int elements_per_thread = (kernel_h * kernel_w + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int kid = tid + i * BLOCK_SIZE;
        if (kid < kernel_h * kernel_w) {
            const int kh = kid / kernel_w;
            const int kw = kid % kernel_w;
            
            const int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            const int w_in = w_out * stride_w - padding_w + kw * dilation_w;
            
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                const int in_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                const int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                reg_sum += input[in_idx] * weight[weight_idx];
            }
        }
    }

    // Store partial sum in shared memory
    shared_mem[tid] = reg_sum;
    
    // Single sync point to ensure all threads have written their partial sums
    __syncthreads();

    // Warp-level reduction without synchronization
    if (tid < WARP_SIZE) {
        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = tid; i < BLOCK_SIZE; i += WARP_SIZE) {
            warp_sum += shared_mem[i];
        }
        
        // Warp shuffle reduction
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        if (lane == 0) {
            if (bias != nullptr) {
                warp_sum += bias[c_out];
            }
            output[out_idx] = warp_sum;
        }
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = groups * weight.size(1);
    const int channels_per_group = out_channels / groups;
    
    const int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    
    const int total_elements = batch_size * out_channels * out_h * out_w;
    const int shared_memory_size = BLOCK_SIZE * sizeof(float);
    
    depthwise_conv2d_kernel<<<total_elements, BLOCK_SIZE, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
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