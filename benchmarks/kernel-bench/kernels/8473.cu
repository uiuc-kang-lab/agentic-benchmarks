#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

__global__ void conv_transpose2d_kernel(float* shared_input, float* shared_weight, float* shared_bias, float* output, int N, int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups) {
    __shared__ float shared_input[256];
    __shared__ float shared_weight[256];
    __shared__ float shared_bias[256];
    int total = N * C_out * H_out * W_out;
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int total = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_idx = blockDim.x * gridDim.x;

    while (idx < total) {
        int w_out = idx % W_out;
        int tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        int out_chan_per_group = C_out / groups;
        int group = c_out / out_chan_per_group;
        int in_chan_start = group * (C_in / groups);
        int in_chan_end = in_chan_start + (C_in / groups);

        float sum = bias ? bias[c_out] : 0.0f;

        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int h_padded = h_out + padding_h - ky * dilation_h;
                int w_padded = w_out + padding_w - kx * dilation_w;
                
                if (h_padded % stride_h == 0 && w_padded % stride_w == 0) {
                    int h_in = h_padded / stride_h;
                    int w_in = w_padded / stride_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        for (int c_in = in_chan_start; c_in < in_chan_end; ++c_in) {
                            int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                            int weight_idx = (((c_in * out_chan_per_group) + 
                                           (c_out % out_chan_per_group)) * kernel_h + ky) * kernel_w + kx;
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
        }

        output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = sum;
        idx += stride_idx;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int C_out = weight.size(1) * groups;

    // Calculate output dimensions
    int H_out = (H_in-1)*stride[0] - 2*padding[0] + dilation[0]*(kernel_h-1) + output_padding[0] + 1;
    int W_out = (W_in-1)*stride[1] - 2*padding[1] + dilation[1]*(kernel_w-1) + output_padding[1] + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    const int threads = 256;
    int total_elements = N * C_out * H_out * W_out;
    int blocks = (total_elements + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward optimized with empty output");
}