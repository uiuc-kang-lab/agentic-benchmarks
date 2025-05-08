#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <int KD, int KH, int KW>
__global__ void conv_transposed_3d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int totalElements,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int D_out, int H_out, int W_out,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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

        #pragma unroll
        for (int r = 0; r < KD; r++) {
            int d_in_calc = d + pad_d - r;
            if (d_in_calc % stride_d != 0 || d_in_calc < 0) continue;
            int d_in = d_in_calc / stride_d;
            if (d_in >= D_in) continue;
            
            #pragma unroll
            for (int s = 0; s < KH; s++) {
                int h_in_calc = h + pad_h - s;
                if (h_in_calc % stride_h != 0 || h_in_calc < 0) continue;
                int h_in = h_in_calc / stride_h;
                if (h_in >= H_in) continue;
                
                #pragma unroll
                for (int t = 0; t < KW; t++) {
                    int w_in_calc = w + pad_w - t;
                    if (w_in_calc % stride_w != 0 || w_in_calc < 0) continue;
                    int w_in = w_in_calc / stride_w;
                    if (w_in >= W_in) continue;

                    for (int c = 0; c < input_channels_per_group; c++) {
                        int actual_c_in = group * input_channels_per_group + c;
                        int input_idx = (((n * C_in + actual_c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        int weight_idx = ((actual_c_in * (C_out/groups) + c_out_in_group) * (KD*KH*KW)) + (r*KH*KW + s*KW + t);
                        out_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        output[idx] = out_val;
        idx += blockDim.x * gridDim.x;
    }
}

template <int KD, int KH, int KW>
torch::Tensor launch_templated_kernel(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.has_value()) {
      bias = bias.value().contiguous();
    }
    
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int kD = KD, kH = KH, kW = KW;
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int output_pad_d = output_padding[0];
    int output_pad_h = output_padding[1];
    int output_pad_w = output_padding[2];
    
    int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + output_pad_d;
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + output_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + output_pad_w;
    
    int output_channels = weight.size(1) * groups;
    
    auto output = torch::zeros({N, output_channels, D_out, H_out, W_out}, input.options());
    
    int totalElements = N * output_channels * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;
    
    conv_transposed_3d_cuda_kernel<KD,KH,KW><<<blocks, threads>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      totalElements,
      N, C_in, D_in, H_in, W_in,
      output_channels,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      D_out, H_out, W_out,
      groups
    );
    
    return output;
}

__global__ void conv_transposed_3d_cuda_generic_kernel(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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
            if (d_in_calc % stride_d != 0 || d_in_calc < 0) continue;
            int d_in = d_in_calc / stride_d;
            if (d_in >= D_in) continue;
            for (int s = 0; s < kH; s++) {
                int h_in_calc = h + pad_h - s;
                if (h_in_calc % stride_h != 0 || h_in_calc < 0) continue;
                int h_in = h_in_calc / stride_h;
                if (h_in >= H_in) continue;
                for (int t = 0; t < kW; t++) {
                    int w_in_calc = w + pad_w - t;
                    if (w_in_calc % stride_w != 0 || w_in_calc < 0) continue;
                    int w_in = w_in_calc / stride_w;
                    if (w_in >= W_in) continue;
                    for (int c = 0; c < input_channels_per_group; c++) {
                        int actual_c_in = group * input_channels_per_group + c;
                        int input_idx = (((n * C_in + actual_c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        int weight_idx = ((actual_c_in * (C_out/groups) + c_out_in_group) * (kD*kH*kW)) + (r*kH*kW + s*kW + t);
                        out_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = out_val;
        idx += blockDim.x * gridDim.x;
    }
}

torch::Tensor launch_generic_kernel(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.has_value()) {
      bias = bias.value().contiguous();
    }
    
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int output_pad_d = output_padding[0];
    int output_pad_h = output_padding[1];
    int output_pad_w = output_padding[2];
    
    int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + output_pad_d;
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + output_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + output_pad_w;
    
    int C_out = weight.size(1) * groups;
    
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());
    int totalElements = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;
    
    conv_transposed_3d_cuda_generic_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
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

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    if (kD == 3 && kH == 3 && kW == 3) {
        return launch_templated_kernel<3,3,3>(input, weight, bias, stride, padding, output_padding, groups);
    } else if (kD == 2 && kH == 2 && kW == 2) {
        return launch_templated_kernel<2,2,2>(input, weight, bias, stride, padding, output_padding, groups);
    } else {
        return launch_generic_kernel(input, weight, bias, stride, padding, output_padding, groups);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d with templated unrolling");
}
