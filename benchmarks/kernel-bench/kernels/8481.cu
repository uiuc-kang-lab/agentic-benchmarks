#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

/* Helper device functions */
__device__ __forceinline__ void decode_index(int idx, int W_out, int H_out, int C_out, 
                                            int& n, int& c_out, int& h_out, int& w_out) {
    w_out = idx % W_out;
    idx /= W_out;
    h_out = idx % H_out;
    idx /= H_out;
    c_out = idx % C_out;
    n = idx / C_out;
}

__device__ __forceinline__ bool valid_input_position(int pos, int stride, int max_dim) {
    return pos >= 0 && pos < max_dim && (pos % stride) == 0;
}

__device__ __forceinline__ void compute_transpose_indices(
    int h_out, int w_out, int kernel_h, int kernel_w,
    int padding_h, int padding_w, int dilation_h, int dilation_w,
    int& k_y, int& k_x, int& h_in, int& w_in, int stride_h, int stride_w, int H_in, int W_in
) {
    h_in = h_out + padding_h - k_y * dilation_h;
    w_in = w_out + padding_w - k_x * dilation_w;
}

/* Main optimized kernel */
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups
) {
    const int total = N * C_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode output indices
    int n, c_out, h_out, w_out;
    decode_index(idx, W_out, H_out, C_out, n, c_out, h_out, w_out);

    // Channel group setup
    const int group_size = C_out / groups;
    const int group = c_out / group_size;
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);

    // Initialize with bias
    float val = bias ? __ldg(&bias[c_out]) : 0.0f;

    for (int k_y = 0; k_y < kernel_h; ++k_y) {
        for (int k_x = 0; k_x < kernel_w; ++k_x) {
            int h_in, w_in;
            compute_transpose_indices(h_out, w_out, kernel_h, kernel_w,
                                     padding_h, padding_w, dilation_h, dilation_w,
                                     k_y, k_x, h_in, w_in, stride_h, stride_w, H_in, W_in);

            if (valid_input_position(h_in, stride_h, H_in) && 
                valid_input_position(w_in, stride_w, W_in)) {
                
                h_in /= stride_h;
                w_in /= stride_w;

                for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
                    const int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    const int weight_idx = ((c_in * group_size + (c_out % group_size)) * kernel_h + k_y) * kernel_w + k_x;
                    
                    val += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                }
            }
        }
    }

    output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    // Dimension calculations
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int C_out = weight.size(1) * groups;

    const int H_out = (H_in - 1) * stride[0] - 2 * padding[0] 
                     + dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
    const int W_out = (W_in - 1) * stride[1] - 2 * padding[1] 
                     + dilation[1] * (kernel_w - 1) + output_padding[1] + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Launch configuration
    const int threads = 256;
    const int blocks = (N * C_out * H_out * W_out + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
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
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D modular optimized (CUDA)");
}