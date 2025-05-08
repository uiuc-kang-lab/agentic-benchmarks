#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int BLOCK_SIZE = 128; // Tested empirically on H100

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * outD * outH * outW;
    if (index >= total_elements) return;

    // Decode output index
    int ow = index % outW;
    int tmp = index / outW;
    int oh = tmp % outH;
    tmp /= outH;
    int od = tmp % outD;
    tmp /= outD;
    int c_out = tmp % C_out;
    int n = tmp / C_out;
    
    int group = c_out / (C_out / groups);
    int oc_in_group = c_out % (C_out / groups);
    int C_in_per_group = C_in / groups;

    float sum = 0.0f;

    for (int kd = 0; kd < kernel_d; ++kd) {
        int d_in = (od + pad_d - kd) / stride_d;
        if ((od + pad_d - kd) % stride_d != 0) continue;
        if (d_in < 0 || d_in >= D_in) continue;

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = (oh + pad_h - kh) / stride_h;
            if ((oh + pad_h - kh) % stride_h != 0) continue;
            if (h_in < 0 || h_in >= H_in) continue;

            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = (ow + pad_w - kw) / stride_w;
                if ((ow + pad_w - kw) % stride_w != 0) continue;
                if (w_in < 0 || w_in >= W_in) continue;

                int c_in_base = group * C_in_per_group;
                for (int c_off = 0; c_off < C_in_per_group; ++c_off) {
                    int c_in = c_in_base + c_off;
                    int input_idx = ((((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in);
                    int weight_idx = ((((c_in * (C_out/groups) + oc_in_group) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias) sum += bias[c_out];
    output[index] = sum;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) CHECK_INPUT(*bias);

    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int outD = (D_in - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0];
    int outH = (H_in - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1];
    int outW = (W_in - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2];
    
    int C_out = weight.size(1) * groups;

    auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());
    int total_elements = output.numel();

    const float* bias_ptr = bias.has_value() ? (*bias).data_ptr<float>() : nullptr;
    
    conv_transpose3d_kernel<<<
        (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        outD, outH, outW,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed Conv3D");
}