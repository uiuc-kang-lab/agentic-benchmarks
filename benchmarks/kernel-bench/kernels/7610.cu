#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int out_depth,
    const int out_height,
    const int out_width
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Calculate output position
    const int total_output_size = batch_size * out_channels * out_depth * out_height * out_width;
    
    for (int idx = gid; idx < total_output_size; idx += gridDim.x * blockDim.x) {
        const int w = idx % out_width;
        const int h = (idx / out_width) % out_height;
        const int d = (idx / (out_width * out_height)) % out_depth;
        const int c = (idx / (out_width * out_height * out_depth)) % out_channels;
        const int b = idx / (out_width * out_height * out_depth * out_channels);
        
        float sum = bias ? bias[c] : 0.0f;
        
        // Align memory access patterns within warps
        #pragma unroll
        for (int ic = 0; ic < in_channels; ++ic) {
            const int in_d_start = d / stride_d - pad_d;
            const int in_h_start = h / stride_h - pad_h;
            const int in_w_start = w / stride_w - pad_w;
            
            for (int kd = 0; kd < kernel_d; ++kd) {
                const int in_d = in_d_start + kd;
                if (in_d >= 0 && in_d < in_depth) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        const int in_h = in_h_start + kh;
                        if (in_h >= 0 && in_h < in_height) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                const int in_w = in_w_start + kw;
                                if (in_w >= 0 && in_w < in_width) {
                                    const int input_idx = ((b * in_channels + ic) * in_depth + in_d) * 
                                                         in_height * in_width + in_h * in_width + in_w;
                                    const int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * 
                                                          kernel_h * kernel_w + kh * kernel_w + kw;
                                    
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    auto output = torch::zeros(
        {x.size(0), weight.size(0),
         (x.size(2) - 1) * stride[0] - 2 * padding[0] + weight.size(2) + output_padding[0],
         (x.size(3) - 1) * stride[1] - 2 * padding[1] + weight.size(3) + output_padding[1],
         (x.size(4) - 1) * stride[2] - 2 * padding[2] + weight.size(4) + output_padding[2]},
        x.options());
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_depth = x.size(2);
    const int in_height = x.size(3);
    const int in_width = x.size(4);
    
    const int out_channels = output.size(1);
    const int out_depth = output.size(2);
    const int out_height = output.size(3);
    const int out_width = output.size(4);
    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const dim3 blocks((output.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 threads(BLOCK_SIZE);
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        out_depth, out_height, out_width
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}