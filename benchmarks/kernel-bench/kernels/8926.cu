#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Device function to compute the flat index for the input tensor
__device__ inline int get_input_index(int n, int c, int in_channels, int in_h, int in_w, int ih, int iw) {
    return n * (in_channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
}

// Device function to compute the flat index for the weight tensor
// Weight layout: [in_channels, out_channels_per_group, kernel_h, kernel_w]
__device__ inline int get_weight_index(int c, int oc, int group, int out_channels_per_group, int kernel_h, int kernel_w, int kh, int kw) {
    return c * (out_channels_per_group * kernel_h * kernel_w) +
           (oc - group * out_channels_per_group) * (kernel_h * kernel_w) +
           kh * kernel_w + kw;
}

// Device function to compute the output value for a given output coordinate
__device__ float compute_output_value(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int n, int oc, int oh, int ow,
    int in_channels, int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups, int in_channels_per_group, int out_channels_per_group) {

    float acc = bias[oc];
    int group = oc / out_channels_per_group;
    
    // Loop over the corresponding input channels for this group
    for (int c = group * in_channels_per_group; c < (group + 1) * in_channels_per_group; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            int h_in_candidate = oh + pad_h - kh * dilation_h;
            if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0) continue;
            int ih = h_in_candidate / stride_h;
            if (ih >= in_h) continue;
            
            for (int kw = 0; kw < kernel_w; kw++) {
                int w_in_candidate = ow + pad_w - kw * dilation_w;
                if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
                int iw = w_in_candidate / stride_w;
                if (iw >= in_w) continue;
                
                int input_idx = get_input_index(n, c, in_channels, in_h, in_w, ih, iw);
                int weight_idx = get_weight_index(c, oc, group, out_channels_per_group, kernel_h, kernel_w, kh, kw);
                acc += x[input_idx] * weight[weight_idx];
            }
        }
    }
    return acc;
}

// Modular CUDA kernel for 2D transposed convolution
__global__ void conv_transpose2d_modular_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int in_channels_per_group,
    int out_channels_per_group) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (index >= total) return;

    // Decode the flat index into (n, oc, oh, ow)
    int ow = index % out_w;
    int tmp = index / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    // Compute the output value using the modular device function
    float out_val = compute_output_value(x, weight, bias, n, oc, oh, ow,
                                          in_channels, in_h, in_w,
                                          kernel_h, kernel_w,
                                          stride_h, stride_w,
                                          pad_h, pad_w,
                                          dilation_h, dilation_w,
                                          groups, in_channels_per_group, out_channels_per_group);

    int out_index = n * (out_channels * out_h * out_w) +
                    oc * (out_h * out_w) +
                    oh * out_w + ow;
    output[out_index] = out_val;
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {

    // Ensure inputs are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    if (bias.has_value() && bias.value().defined())
        bias = bias.value().contiguous();

    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({out_channels}, weight.options());
    }

    auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

    int in_channels_per_group = in_channels / groups;
    int total_threads = batch * out_channels * out_h * out_w;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    conv_transpose2d_modular_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels_per_group,
        out_channels_per_group
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular 2D Transposed Convolution (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
