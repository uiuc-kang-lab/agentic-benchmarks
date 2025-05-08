#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define KERNEL_SIZE 3
__constant__ float weight_const[64 * 64 * 3 * 3 * 3];  // Constant memory for weights

template<int KERNEL_D, int KERNEL_H, int KERNEL_W>
__global__ void conv3d_unrolled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int stride_size = gridDim.x * blockDim.x;

    #pragma unroll 1
    for (int idx = tid; idx < total_elements; idx += stride_size) {
        const int w_out = idx % out_width;
        const int h_out = (idx / out_width) % out_height;
        const int d_out = (idx / (out_width * out_height)) % out_depth;
        const int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
        const int b = idx / (out_width * out_height * out_depth * out_channels);

        float sum = 0.0f;
        const int group = c_out / (out_channels / groups);
        const int in_channels_per_group = in_channels / groups;

        #pragma unroll 1
        for (int ic = 0; ic < in_channels_per_group; ic++) {
            const int in_c = group * in_channels_per_group + ic;
            
            // Fully unroll the kernel loops since we know the dimensions are small (3x3x3)
            #pragma unroll
            for (int kd = 0; kd < KERNEL_D; kd++) {
                const int d_in = d_out * stride - padding + kd * dilation;
                if (d_in >= 0 && d_in < in_depth) {
                    
                    #pragma unroll
                    for (int kh = 0; kh < KERNEL_H; kh++) {
                        const int h_in = h_out * stride - padding + kh * dilation;
                        if (h_in >= 0 && h_in < in_height) {
                            
                            #pragma unroll
                            for (int kw = 0; kw < KERNEL_W; kw++) {
                                const int w_in = w_out * stride - padding + kw * dilation;
                                if (w_in >= 0 && w_in < in_width) {
                                    const int input_idx = ((b * in_channels + in_c) * in_depth + d_in) * 
                                                        in_height * in_width + h_in * in_width + w_in;
                                    const int weight_idx = (((c_out * in_channels_per_group) + ic) * 
                                                         KERNEL_D + kd) * KERNEL_H * KERNEL_W + 
                                                         kh * KERNEL_W + kw;
                                    
                                    sum += input[input_idx] * weight_const[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        output[idx] = sum;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

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

    // Copy weights to constant memory
    cudaMemcpyToSymbol(weight_const, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv3d_unrolled_kernel<KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE>
        <<<num_blocks, BLOCK_SIZE>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            batch_size,
            in_channels,
            out_channels,
            in_depth,
            in_height,
            in_width,
            out_depth,
            out_height,
            out_width,
            stride,
            padding,
            dilation,
            groups
        );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with aggressive unrolling (CUDA)");
}