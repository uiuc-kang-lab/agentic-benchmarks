#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined constant memory array to store frequently accessed read-only parameters
// Layout (16 ints): [0]: stride_d, [1]: stride_h, [2]: stride_w,
// [3]: pad_d, [4]: pad_h, [5]: pad_w,
// [6]: N, [7]: C_in, [8]: D_in, [9]: H_in, [10]: W_in,
// [11]: C_out, [12]: D_out, [13]: H_out, [14]: W_out, [15]: groups
__constant__ int c_params[16];

// Optimized CUDA kernel using constant memory parameters
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
        // Retrieve output dimensions and groups from constant memory
        int W_out = c_params[14];
        int H_out = c_params[13];
        int D_out = c_params[12];
        int C_out = c_params[11];
        int groups = c_params[15];

        // Decode flat index into (n, c_out, d, h, w)
        int w = idx % W_out;
        int tmp = idx / W_out;
        int h = tmp % H_out;
        tmp /= H_out;
        int d = tmp % D_out;
        tmp /= D_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;
        
        int out_channels_per_group = C_out / groups;
        int group = c_out / out_channels_per_group;
        int c_out_in_group = c_out - group * out_channels_per_group;
        int C_in = c_params[7];
        int input_channels_per_group = C_in / groups;
        
        float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        // Retrieve stride and padding parameters from constant memory
        int stride_d = c_params[0], stride_h = c_params[1], stride_w = c_params[2];
        int pad_d = c_params[3], pad_h = c_params[4], pad_w = c_params[5];
        int D_in = c_params[8], H_in = c_params[9], W_in = c_params[10];
        
        // Iterate over kernel dimensions
        for (int r = 0; r < kD; r++) {
            int d_in_calc = d + pad_d - r;
            if (d_in_calc < 0 || (d_in_calc % stride_d) != 0) continue;
            int d_in = d_in_calc / stride_d;
            if (d_in < 0 || d_in >= D_in) continue;
            
            for (int s = 0; s < kH; s++) {
                int h_in_calc = h + pad_h - s;
                if (h_in_calc < 0 || (h_in_calc % stride_h) != 0) continue;
                int h_in = h_in_calc / stride_h;
                if (h_in < 0 || h_in >= H_in) continue;
                
                for (int t = 0; t < kW; t++) {
                    int w_in_calc = w + pad_w - t;
                    if (w_in_calc < 0 || (w_in_calc % stride_w) != 0) continue;
                    int w_in = w_in_calc / stride_w;
                    if (w_in < 0 || w_in >= W_in) continue;
                    
                    for (int c = 0; c < input_channels_per_group; c++) {
                        int actual_c_in = group * input_channels_per_group + c;
                        int input_index = (((n * C_in + actual_c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        float in_val = input[input_index];
                        
                        int weight_index = ((actual_c_in * out_channels_per_group + c_out_in_group) * (kD * kH * kW))
                                           + (r * kH * kW + s * kW + t);
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

// Forward function with cached constant memory parameters
// It caches the constant parameters across calls to avoid redundant cudaMemcpyToSymbol calls
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Get input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    
    // Get kernel dimensions
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    
    const int output_channels_per_group = weight.size(1);
    const int C_out = output_channels_per_group * groups;
    
    const int D_out = (D_in - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0];
    const int H_out = (H_in - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int W_out = (W_in - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];
    
    // Prepare combined constant parameters array (16 ints)
    // [0-2]: stride (d, h, w), [3-5]: padding (d, h, w),
    // [6]: N, [7]: C_in, [8]: D_in, [9]: H_in, [10]: W_in,
    // [11]: C_out, [12]: D_out, [13]: H_out, [14]: W_out, [15]: groups
    int h_params[16];
    h_params[0] = static_cast<int>(stride[0]);
    h_params[1] = static_cast<int>(stride[1]);
    h_params[2] = static_cast<int>(stride[2]);
    h_params[3] = static_cast<int>(padding[0]);
    h_params[4] = static_cast<int>(padding[1]);
    h_params[5] = static_cast<int>(padding[2]);
    h_params[6] = N;
    h_params[7] = C_in;
    h_params[8] = D_in;
    h_params[9] = H_in;
    h_params[10] = W_in;
    h_params[11] = C_out;
    h_params[12] = D_out;
    h_params[13] = H_out;
    h_params[14] = W_out;
    h_params[15] = static_cast<int>(groups);
    
    // Cache constant parameters to avoid redundant copies
    static int last_params[16] = { -1 };
    static bool first_call = true;
    bool update = first_call;
    if (!update) {
        for (int i = 0; i < 16; i++) {
            if (last_params[i] != h_params[i]) {
                update = true;
                break;
            }
        }
    }
    if (update) {
        cudaMemcpyToSymbol(c_params, h_params, sizeof(int) * 16);
        for (int i = 0; i < 16; i++) {
            last_params[i] = h_params[i];
        }
        first_call = false;
    }
    
    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());
    
    int totalElements = N * C_out * D_out * H_out * W_out;
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    
    conv_transposed_3d_cuda_kernel<<<gridSize, blockSize, 0, cudaStreamDefault>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        totalElements,
        kD, kH, kW
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with cached constant memory",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
