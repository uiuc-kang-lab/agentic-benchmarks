#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper function: parse an int or a sequence into exactly 2 dimensions
inline std::vector<int64_t> parseDim(py::object obj) {
    std::vector<int64_t> dims;
    if (py::isinstance<py::int_>(obj)) {
        int64_t val = obj.cast<int64_t>();
        dims.push_back(val);
        dims.push_back(val);
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            dims.push_back(py::cast<int64_t>(item));
        }
        if (dims.size() == 1) {
            dims.push_back(dims[0]);
        }
        if (dims.size() != 2) {
            throw std::runtime_error("Expected exactly 2 integers for 2D operation.");
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return dims;
}

// Optimized CUDA kernel for ConvTranspose2d using warp-level reduction
// Each warp computes one output element. The workload is distributed over the warp lanes,
// which then perform an efficient reduction using __shfl_down_sync for summation.
__global__ void conv_transposed2d_optimized_kernel(
    const float* __restrict__ input,
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

    int lane = threadIdx.x % warpSize;  // lane within the warp

    // Decode global_warp_id to (n, oc, oh, ow)
    int tmp = global_warp_id;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int oc = tmp % out_channels;
    tmp /= out_channels;
    int n = tmp;

    float sum = 0.0f;

    // Determine the group from oc and corresponding range of input channels
    int group = oc / out_channels_per_group;
    int start_ic = group * in_channels_per_group;

    // Total iterations for reduction = number of input channels in this group * kernel spatial size
    int total_iters = in_channels_per_group * kernel_h * kernel_w;

    // Each warp lane processes a subset of the iterations
    for (int iter = lane; iter < total_iters; iter += warpSize) {
        int ic_offset = iter / (kernel_h * kernel_w);
        int rem = iter % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;

        int ic = start_ic + ic_offset;

        // Compute the corresponding input spatial location
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
        sum += input[input_index] * weight[weight_index];
    }

    // Warp-level reduction: accumulate partial sums from all lanes in the warp
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // The first lane in each warp writes the computed output element
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_index = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
        output[output_index] = sum;
    }
}

// Forward wrapper that prepares input tensors and launches the optimized kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    // Parse stride, padding, and output_padding into 2D parameters
    auto stride_vec = parseDim(stride);
    auto padding_vec = parseDim(padding);
    auto output_padding_vec = parseDim(output_padding);

    int stride_h = stride_vec[0], stride_w = stride_vec[1];
    int pad_h = padding_vec[0], pad_w = padding_vec[1];
    int output_pad_h = output_padding_vec[0], output_pad_w = output_padding_vec[1];

    // Get input dimensions: (N, C, H, W)
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    // Weight dimensions: (in_channels, out_channels/groups, kernel_h, kernel_w)
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    // Compute output spatial dimensions for ConvTranspose2d
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    // Channels per group
    int in_channels_per_group = in_channels / groups;

    // Determine grid configuration: one warp computes one output element
    int total_outputs = batch_size * out_channels * out_h * out_w;
    const int warpSize = 32;
    int warps_per_block = 8; // Using 8 warps per block (256 threads per block)
    int threads_per_block = warps_per_block * warpSize;
    int blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value().contiguous();
    }

    conv_transposed2d_optimized_kernel<<<blocks, threads_per_block>>>(
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
    m.def("forward", &forward, "Optimized ConvTranspose2D forward with warp-level reduction and coalesced memory access",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
