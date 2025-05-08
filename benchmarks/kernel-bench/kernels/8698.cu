#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__constant__ float c_weights[32768];  // 32KB capacity

__global__ void conv_transpose3d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int in_channels,
    const int in_d,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int groups) {

    const int total_output = batch_size * out_channels * out_d * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    // Output tensor indexing
    int n = idx / (out_channels * out_d * out_h * out_w);
    int c_out = (idx / (out_d * out_h * out_w)) % out_channels;
    int d_out = (idx / (out_h * out_w)) % out_d;
    int h_out = (idx / out_w) % out_h;
    int w_out = idx % out_w;

    int g = c_out / (out_channels / groups);
    float sum = 0.0f;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int d_in = (d_out - kd + pad_d) / stride_d;
                int h_in = (h_out - kh + pad_h) / stride_h;
                int w_in = (w_out - kw + pad_w) / stride_w;

                if ((d_out - kd + pad_d) % stride_d != 0) continue;
                if ((h_out - kh + pad_h) % stride_h != 0) continue;
                if ((w_out - kw + pad_w) % stride_w != 0) continue;

                if (d_in < 0 || d_in >= in_d) continue;
                if (h_in < 0 || h_in >= in_h) continue;
                if (w_in < 0 || w_in >= in_w) continue;

                int c_in_start = g * (in_channels / groups);
                int c_in_end = c_in_start + (in_channels / groups);

                for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
                    int input_idx = n * in_channels * in_d * in_h * in_w +
                                   c_in * in_d * in_h * in_w +
                                   d_in * in_h * in_w +
                                   h_in * in_w +
                                   w_in;

                    int weight_idx = c_out * (in_channels/groups) * kernel_d * kernel_h * kernel_w +
                                   (c_in - c_in_start) * kernel_d * kernel_h * kernel_w +
                                   kd * kernel_h * kernel_w +
                                   kh * kernel_w +
                                   kw;

                    sum += input[input_idx] * c_weights[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    
    // Copy weights to constant memory
    size_t weight_bytes = weight.numel() * sizeof(float);
    TORCH_CHECK(weight_bytes <= sizeof(c_weights), "Weights exceed constant memory capacity");
    cudaMemcpyToSymbol(c_weights, weight.data_ptr<float>(), weight_bytes, 0, cudaMemcpyDeviceToDevice);

    // Calculate output dimensions
    int64_t out_d = (x.size(2)-1)*stride[0] - 2*padding[0] + weight.size(2) + output_padding[0];
    int64_t out_h = (x.size(3)-1)*stride[1] - 2*padding[1] + weight.size(3) + output_padding[1];
    int64_t out_w = (x.size(4)-1)*stride[2] - 2*padding[2] + weight.size(4) + output_padding[2];

    auto output = torch::zeros({x.size(0), weight.size(1)*groups, out_d, out_h, out_w}, x.options());

    // Kernel launch parameters
    int64_t num_elements = output.numel();
    dim3 blocks((num_elements + 255) / 256);
    dim3 threads(256);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.size(0), x.size(1),
        x.size(2), x.size(3), x.size(4),
        output.size(1), output.size(2), output.size(3), output.size(4),
        weight.size(2), weight.size(3), weight.size(4),
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        groups
    );

    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        CHECK_INPUT(bias_tensor);
        output += bias_tensor.view({1, -1, 1, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed Conv3D with constant memory weights");
}