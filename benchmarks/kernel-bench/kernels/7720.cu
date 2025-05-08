#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized block dimensions for H100
#define BLOCK_SIZE 1024
#define TILE_SIZE 8

// Constant memory for weights
// Removed constant memory declaration; using global memory for weights

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
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation
) {
    // Calculate output position
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const int oc = blockIdx.x;
    const int batch_id = blockIdx.y;
    
    // Use thread ID to compute 3D position
    const int total_elements = out_depth * out_height * out_width;
    
    // Each thread processes multiple elements
    for (int idx = tid; idx < total_elements; idx += block_size) {
        const int od = idx / (out_height * out_width);
        const int tmp = idx % (out_height * out_width);
        const int oh = tmp / out_width;
        const int ow = tmp % out_width;
        
        float sum = 0.0f;
        
        #pragma unroll
        for (int ic = 0; ic < in_channels; ++ic) {
            #pragma unroll
            for (int kd = 0; kd < kernel_d; ++kd) {
                const int id = od * stride - padding + kd * dilation;
                if (id >= 0 && id < in_depth) {
                    #pragma unroll
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        const int ih = oh * stride - padding + kh * dilation;
                        if (ih >= 0 && ih < in_height) {
                            #pragma unroll
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                const int iw = ow * stride - padding + kw * dilation;
                                if (iw >= 0 && iw < in_width) {
                                    const int input_idx = ((batch_id * in_channels + ic) * in_depth + id) *
                                                         in_height * in_width + ih * in_width + iw;
                                    const int weight_idx = ((oc * in_channels + ic) * kernel_d + kd) *
                                                          kernel_h * kernel_w + kh * kernel_w + kw;
                                    sum += input[input_idx] * const_weights[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        const int output_idx = ((batch_id * out_channels + oc) * out_depth + od) *
                               out_height * out_width + oh * out_width + ow;
        output[output_idx] = sum;
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
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width},
                           input.options());
    
    // Copy weights to constant memory
    cudaMemcpyToSymbol(const_weights, weight.data_ptr<float>(), weight.numel() * sizeof(float));

    // Launch kernel with optimized configuration
    dim3 grid(out_channels, batch_size);
    int num_threads = BLOCK_SIZE;
    
    conv3d_optimized_kernel<<<grid, num_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward (CUDA)");
}
