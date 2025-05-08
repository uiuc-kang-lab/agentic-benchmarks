#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__global__ void conv3d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int dilation
) {
    // Each block handles one output channel for one batch
    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    
    // Thread indices
    const int tid = threadIdx.x;
    const int thread_count = blockDim.x;
    
    // Pre-compute output size
    const int out_size = out_depth * out_height * out_width;
    
    // Load bias if present
    const float bias_val = bias != nullptr ? bias[out_channel] : 0.0f;
    extern __shared__ float weight_tile[];
    const int weight_tile_size = in_channels * kernel_d * kernel_h * kernel_w;
    for (int i = tid; i < weight_tile_size; i += thread_count) {
        weight_tile[i] = weight[out_channel * weight_tile_size + i];
    }
    __syncthreads();
    
    // Each thread processes multiple output elements
    for (int out_idx = tid; out_idx < out_size; out_idx += thread_count) {
        // Convert linear index to 3D coordinates
        const int od = out_idx / (out_height * out_width);
        const int tmp = out_idx % (out_height * out_width);
        const int oh = tmp / out_width;
        const int ow = tmp % out_width;
        
        float sum = 0.0f;
        
        // Input starting position
        const int id_start = od * stride - padding;
        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;
        
        // Compute convolution for this output element
        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ++ic) {
            #pragma unroll 2
            for (int kd = 0; kd < kernel_d; ++kd) {
                const int id = id_start + kd * dilation;
                if (id >= 0 && id < in_depth) {
                    #pragma unroll 2
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        const int ih = ih_start + kh * dilation;
                        if (ih >= 0 && ih < in_height) {
                            #pragma unroll 2
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                const int iw = iw_start + kw * dilation;
                                if (iw >= 0 && iw < in_width) {
                                    // Input index
                                    const int in_idx = ((batch_idx * in_channels + ic) * in_depth + id) * 
                                                     in_height * in_width + ih * in_width + iw;
                                    
                                    // Weight index
                                    const int w_idx = ((out_channel * in_channels + ic) * kernel_d + kd) *
                                                    kernel_h * kernel_w + kh * kernel_w + kw;
                                    
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Write output
        const int out_linear_idx = ((batch_idx * out_channels + out_channel) * out_depth + od) *
                                 out_height * out_width + oh * out_width + ow;
        output[out_linear_idx] = sum + bias_val;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "Only groups=1 is supported");
    
    auto bias = bias_opt.value_or(at::Tensor());
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    // Grid and block configuration
    const int threads_per_block = BLOCK_SIZE * WARP_SIZE;
    const dim3 grid(batch_size, out_channels);
    
    conv3d_optimized_kernel<<<grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride, padding, dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward (CUDA)");
}