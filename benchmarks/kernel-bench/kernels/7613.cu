#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 4

__global__ void conv_transposed_3d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int D_out, const int H_out, const int W_out,
    const int kD, const int kH, const int kW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups
) {
    __shared__ float shared_weight[BLOCK_DIM_Z][BLOCK_DIM_Y][BLOCK_DIM_X];
    
    const int w_out = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    const int h_out = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    const int d_out = blockIdx.z * BLOCK_DIM_Z + threadIdx.z;
    
    if (w_out >= W_out || h_out >= H_out || d_out >= D_out) return;
    
    const int batch_size = (N * C_out + gridDim.x - 1) / gridDim.x;
    
    for (int batch_c = 0; batch_c < batch_size; batch_c++) {
        const int n = (batch_c * gridDim.x + blockIdx.x) / C_out;
        const int c_out = (batch_c * gridDim.x + blockIdx.x) % C_out;
        
        if (n >= N) continue;
        
        const int group = c_out / (C_out / groups);
        const int c_out_in_group = c_out % (C_out / groups);
        const int input_channels_per_group = C_in / groups;
        
        float result = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        for (int c_in_offset = 0; c_in_offset < input_channels_per_group; c_in_offset += BLOCK_DIM_X) {
            const int c_in = c_in_offset + threadIdx.x;
            
            if (c_in < input_channels_per_group) {
                const int actual_c_in = group * input_channels_per_group + c_in;
                
                for (int k_d = 0; k_d < kD; k_d++) {
                    const int d_in_calc = d_out + pad_d - k_d;
                    if (d_in_calc % stride_d == 0) {
                        const int d_in = d_in_calc / stride_d;
                        
                        if (d_in >= 0 && d_in < D_in) {
                            for (int k_h = 0; k_h < kH; k_h++) {
                                const int h_in_calc = h_out + pad_h - k_h;
                                if (h_in_calc % stride_h == 0) {
                                    const int h_in = h_in_calc / stride_h;
                                    
                                    if (h_in >= 0 && h_in < H_in) {
                                        for (int k_w = 0; k_w < kW; k_w++) {
                                            const int w_in_calc = w_out + pad_w - k_w;
                                            if (w_in_calc % stride_w == 0) {
                                                const int w_in = w_in_calc / stride_w;
                                                
                                                if (w_in >= 0 && w_in < W_in) {
                                                    const int input_idx = ((n * C_in + actual_c_in) * D_in + d_in) * H_in * W_in +
                                                                         h_in * W_in + w_in;
                                                    const int weight_idx = ((actual_c_in * (C_out/groups) + c_out_in_group) * kD * kH * kW) +
                                                                          (k_d * kH * kW + k_h * kW + k_w);
                                                    
                                                    result += input[input_idx] * weight[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        
        const int out_idx = ((n * C_out + c_out) * D_out + d_out) * H_out * W_out +
                           h_out * W_out + w_out;
        output[out_idx] = result;
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
    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto D_in = input.size(2);
    const auto H_in = input.size(3);
    const auto W_in = input.size(4);

    const auto kD = weight.size(2);
    const auto kH = weight.size(3);
    const auto kW = weight.size(4);
    
    const auto C_out = weight.size(1) * groups;

    const int D_out = (D_in - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0];
    const int H_out = (H_in - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int W_out = (W_in - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
    dim3 blocks(
        (W_out + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (H_out + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y,
        (D_out + BLOCK_DIM_Z - 1) / BLOCK_DIM_Z
    );

    conv_transposed_3d_cuda_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}