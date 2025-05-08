#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using shared memory for weights
template<int BLOCK_SIZE = 128, int KERNEL_CACHE_SIZE = 128>
__global__ void conv_transposed_3d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int totalElements,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int D_out, int H_out, int W_out,
    int groups
) {
    // Shared memory for caching weight values
    __shared__ float weight_cache[KERNEL_CACHE_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    
    // Collaborative loading of weights into shared memory
    for (int i = tid; i < KERNEL_CACHE_SIZE; i += BLOCK_SIZE) {
        weight_cache[i] = weight[i];
    }
    __syncthreads();  // Single sync point after shared memory initialization
    
    while (idx < totalElements) {
        int w = idx % W_out;
        int tmp = idx / W_out;
        int h = tmp % H_out;
        tmp /= H_out;
        int d = tmp % D_out;
        tmp /= D_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        int output_channels_per_group = C_out / groups;
        int group = c_out / output_channels_per_group;
        int c_out_in_group = c_out - group * output_channels_per_group;
        int input_channels_per_group = C_in / groups;

        float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

        for (int r = 0; r < kD; r++) {
            int d_in_calc = d + pad_d - r;
            if (d_in_calc % stride_d != 0) continue;
            int d_in = d_in_calc / stride_d;
            if (d_in < 0 || d_in >= D_in) continue;
            
            for (int s = 0; s < kH; s++) {
                int h_in_calc = h + pad_h - s;
                if (h_in_calc % stride_h != 0) continue;
                int h_in = h_in_calc / stride_h;
                if (h_in < 0 || h_in >= H_in) continue;
                
                for (int t = 0; t < kW; t++) {
                    int w_in_calc = w + pad_w - t;
                    if (w_in_calc % stride_w != 0) continue;
                    int w_in = w_in_calc / stride_w;
                    if (w_in < 0 || w_in >= W_in) continue;

                    for (int c = 0; c < input_channels_per_group; c++) {
                        int actual_c_in = group * input_channels_per_group + c;
                        int input_index = (((n * C_in + actual_c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        float in_val = input[input_index];
                        
                        int weight_index = ((actual_c_in * output_channels_per_group + c_out_in_group) * (kD * kH * kW))
                                         + (r * kH * kW + s * kW + t);
                        float w_val = (weight_index < KERNEL_CACHE_SIZE) ? 
                                     weight_cache[weight_index] : weight[weight_index];
                        
                        out_val += in_val * w_val;
                    }
                }
            }
        }
        
        output[idx] = out_val;
        idx += blockDim.x * gridDim.x;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    
    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];

    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];

    const int output_channels_per_group = weight.size(1);
    const int C_out = output_channels_per_group * groups;
    const int D_out = (D_in - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0];
    const int H_out = (H_in - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int W_out = (W_in - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const int blockSize = 256;
    const int totalElements = N * C_out * D_out * H_out * W_out;
    const int gridSize = (totalElements + blockSize - 1) / blockSize;

    const float *input_ptr = input.data_ptr<float>();
    const float *weight_ptr = weight.data_ptr<float>();
    const float *bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float *output_ptr = output.data_ptr<float>();

    conv_transposed_3d_cuda_kernel<256, 32><<<gridSize, blockSize>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        totalElements,
        N, C_in, D_in, H_in, W_in,
        C_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        D_out, H_out, W_out,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with shared memory optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}