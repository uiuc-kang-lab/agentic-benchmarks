#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper to parse int or sequence of ints
inline std::vector<int64_t> parseIntArrayRef(const py::object &obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<int64_t>());
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

// Kernel using warp-level primitives for reduction
__global__ void conv_transposed2d_warp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int in_channels_per_group,
    int out_channels_per_group
) {
    const int warpSize = 32;
    // Each warp computes one output element
    int warps_per_block = blockDim.x / warpSize;
    int warp_id_in_block = threadIdx.x / warpSize;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;

    // Total number of output elements
    int total_outputs = batch_size * out_channels * out_h * out_w;
    if (global_warp_id >= total_outputs) return;

    // Lane index within the warp
    int lane = threadIdx.x % warpSize;

    // Decode the output index into (n, oc, oh, ow)
    int tmp = global_warp_id;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int oc = tmp % out_channels;
    tmp /= out_channels;
    int n = tmp;

    // Identify the group and corresponding input channels
    int group = oc / out_channels_per_group;
    int start_ic = group * in_channels_per_group;

    // Total iterations over (ic, kh, kw) for the current group
    int total_iters = in_channels_per_group * kernel_h * kernel_w;
    float sum = 0.0f; 
    #pragma unroll
    for (int iter = lane; iter < total_iters; iter += warpSize) {

    // Distribute the iterations among warp lanes
    for (int iter = lane; iter < total_iters; iter += warpSize) {
        int ic_offset = iter / (kernel_h * kernel_w);
        int rem = iter % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;
        
        int ic = start_ic + ic_offset;

        int i_h = oh + pad_h - kh;
        if (i_h % stride_h != 0) continue;
        int i_h_div = i_h / stride_h;
        if (i_h_div < 0 || i_h_div >= in_h) continue;

        int i_w = ow + pad_w - kw;
        if (i_w % stride_w != 0) continue;
        int i_w_div = i_w / stride_w;
        if (i_w_div < 0 || i_w_div >= in_w) continue;

        int input_index = ((n * in_channels + ic) * in_h + i_h_div) * in_w + i_w_div;
        int weight_index = ((ic) * out_channels_per_group + (oc % out_channels_per_group)) * (kernel_h * kernel_w) + kh * kernel_w + kw;
        
        sum += x[input_index] * weight[weight_index];
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // First thread in the warp writes the result
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_index = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
        output[output_index] = sum;
    }
}

// Forward function wrapping the kernel launch
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);
    
    // Ensure 2D parameters (duplicate if necessary)
    if (stride_vec.size() < 2) stride_vec.resize(2, stride_vec[0]);
    if (padding_vec.size() < 2) padding_vec.resize(2, padding_vec[0]);
    if (output_padding_vec.size() < 2) output_padding_vec.resize(2, output_padding_vec[0]);

    int stride_h = stride_vec[0];
    int stride_w = stride_vec[1];
    int pad_h = padding_vec[0];
    int pad_w = padding_vec[1];
    int output_pad_h = output_padding_vec[0];
    int output_pad_w = output_padding_vec[1];
    
    // Input dimensions: [batch_size, in_channels, in_h, in_w]
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    
    // Weight dimensions assumed: [in_channels, out_channels/groups, kernel_h, kernel_w]
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    
    // Compute output dimensions for conv_transpose2d
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    
    int in_channels_per_group = in_channels / groups;
    int total_outputs = batch_size * out_channels * out_h * out_w;
    
    // Configure kernel launch: one warp per output element
    int warpSize = 32;
    int warps_per_block = 8; // e.g., 8 warps per block -> 256 threads per block
    int threads_per_block = warps_per_block * warpSize;
    int blocks = (total_outputs + warps_per_block - 1) / warps_per_block;
    
    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value().contiguous();
    }
    
    conv_transposed2d_warp_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        in_channels_per_group,
        out_channels_per_group
    );
    cudaDeviceSynchronize();
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with warp-level reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
