#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory declarations for frequently accessed parameters
__constant__ int c_stride[3];
__constant__ int c_padding[3];
__constant__ int c_dims[10];  // Store N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out, groups

// Optimized CUDA kernel using constant memory
__global__ void conv_transposed_3d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int totalElements,
    int kD, int kH, int kW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < totalElements) {
        // Decode flat index using constant memory dimensions
        int w = idx % c_dims[8];  // W_out
        int tmp = idx / c_dims[8];
        int h = tmp % c_dims[7];  // H_out
        tmp /= c_dims[7];
        int d = tmp % c_dims[6];  // D_out
        tmp /= c_dims[6];
        int c_out = tmp % c_dims[5];  // C_out
        int n = tmp / c_dims[5];

        // Group calculations using constant memory
        int output_channels_per_group = c_dims[5] / c_dims[9];  // C_out / groups
        int group = c_out / output_channels_per_group;
        int c_out_in_group = c_out - group * output_channels_per_group;
        int input_channels_per_group = c_dims[1] / c_dims[9];  // C_in / groups

        // Initialize accumulator
        float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

        for (int r = 0; r < kD; r++) {
            int d_in_calc = d + c_padding[0] - r;
            if (d_in_calc % c_stride[0] != 0) continue;
            int d_in = d_in_calc / c_stride[0];
            if (d_in < 0 || d_in >= c_dims[2]) continue;
            
            for (int s = 0; s < kH; s++) {
                int h_in_calc = h + c_padding[1] - s;
                if (h_in_calc % c_stride[1] != 0) continue;
                int h_in = h_in_calc / c_stride[1];
                if (h_in < 0 || h_in >= c_dims[3]) continue;
                
                for (int t = 0; t < kW; t++) {
                    int w_in_calc = w + c_padding[2] - t;
                    if (w_in_calc % c_stride[2] != 0) continue;
                    int w_in = w_in_calc / c_stride[2];
                    if (w_in < 0 || w_in >= c_dims[4]) continue;

                    for (int c = 0; c < input_channels_per_group; c++) {
                        int actual_c_in = group * input_channels_per_group + c;
                        
                        // Calculate indices using constant memory dimensions
                        int input_index = (((n * c_dims[1] + actual_c_in) * c_dims[2] + d_in) 
                                         * c_dims[3] + h_in) * c_dims[4] + w_in;
                        float in_val = input[input_index];
                        
                        int weight_index = ((actual_c_in * output_channels_per_group + c_out_in_group) 
                                          * (kD * kH * kW)) + (r * kH * kW + s * kW + t);
                        float w_val = weight[weight_index];
                        
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
    // Get dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    
    // Calculate output dimensions
    const int output_channels_per_group = weight.size(1);
    const int C_out = output_channels_per_group * groups;
    const int D_out = (D_in - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0];
    const int H_out = (H_in - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int W_out = (W_in - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    // Copy constant parameters to device
    int h_stride[3] = {static_cast<int>(stride[0]), static_cast<int>(stride[1]), static_cast<int>(stride[2])};
    int h_padding[3] = {static_cast<int>(padding[0]), static_cast<int>(padding[1]), static_cast<int>(padding[2])};
    int h_dims[10] = {N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out, static_cast<int>(groups)};
    
    cudaMemcpyToSymbol(c_stride, h_stride, sizeof(int) * 3);
    cudaMemcpyToSymbol(c_padding, h_padding, sizeof(int) * 3);
    cudaMemcpyToSymbol(c_dims, h_dims, sizeof(int) * 10);

    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    // Launch configuration
    int totalElements = N * C_out * D_out * H_out * W_out;
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;

    // Get raw pointers
    const float *input_ptr = input.data_ptr<float>();
    const float *weight_ptr = weight.data_ptr<float>();
    const float *bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float *output_ptr = output.data_ptr<float>();

    // Launch kernel
    conv_transposed_3d_cuda_kernel<<<gridSize, blockSize>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        totalElements,
        kD, kH, kW
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with constant memory optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}