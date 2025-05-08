#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define SMALL_KERNEL_THRESHOLD 5
#define SHARED_BLOCK_SIZE 64

__global__ void depthwise_conv2d_kernel_2d(
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
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int b = blockIdx.z / out_channels;
    const int c_out = blockIdx.z % out_channels;

    if (x >= out_w || y >= out_h || b >= batch_size) return;

    const int g = c_out / channels_per_group;
    const int m = c_out % channels_per_group;

    float sum = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_in = y * stride_h - padding_h + kh * dilation_h;
        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int w_in = x * stride_w - padding_w + kw * dilation_w;
            
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                const int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                const int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[((b * out_channels + c_out) * out_h + y) * out_w + x] = sum;
}

__global__ void depthwise_conv2d_kernel_shared(
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
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int index = blockIdx.x;
    
    if (index >= batch_size * out_channels * out_h * out_w) return;
    
    const int w_out = index % out_w;
    const int h_out = (index / out_w) % out_h;
    const int c_out = (index / (out_w * out_h)) % out_channels;
    const int b = index / (out_channels * out_h * out_w);
    const int g = c_out / channels_per_group;
    
    float partial_sum = 0.0f;
    const int kernel_size = kernel_h * kernel_w;
    
    for (int k = tid; k < kernel_size; k += blockDim.x) {
        const int kh = k / kernel_w;
        const int kw = k % kernel_w;
        const int h_in = h_out * stride_h - padding_h + kh * dilation_h;
        const int w_in = w_out * stride_w - padding_w + kw * dilation_w;
        
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            const int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            const int weight_idx = ((g * channels_per_group + (c_out % channels_per_group)) * kernel_h + kh) * kernel_w + kw;
            partial_sum += input[input_idx] * weight[weight_idx];
        }
    }
    
    sdata[tid] = partial_sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        
        if (tid == 0) {
            if (bias != nullptr) {
                val += bias[c_out];
            }
            output[index] = val;
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
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    if (kernel_h * kernel_w <= SMALL_KERNEL_THRESHOLD * SMALL_KERNEL_THRESHOLD) {
        dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 blocks(
            (out_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
            (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
            batch_size * out_channels
        );

        depthwise_conv2d_kernel_2d<<<blocks, threads>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), batch_size, in_channels, in_h, in_w,
            out_channels, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, groups, channels_per_group
        );
    } else {
        const int total_outputs = batch_size * out_channels * out_h * out_w;
        const int blockSize = SHARED_BLOCK_SIZE;
        const int gridSize = total_outputs;
        const size_t shared_mem = blockSize * sizeof(float);

        depthwise_conv2d_kernel_shared<<<gridSize, blockSize, shared_mem>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), batch_size, in_channels, in_h, in_w,
            out_channels, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w, groups, channels_per_group
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Depthwise Conv2D forward (CUDA)");
}