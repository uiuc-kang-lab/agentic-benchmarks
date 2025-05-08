#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Utility function to parse int or sequence of ints
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

// Optimized CUDA kernel for ConvTranspose2d using shared memory reduction and warp-level primitives
// Each block computes one output element. Threads in the block cooperatively reduce the contributions
// over the in_channels_per_group dimension (and kernel window) using shared memory and __shfl_down_sync.

__global__ void conv_transpose2d_shared_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int h_in,
    const int w_in,
    const int out_channels,
    const int h_out,
    const int w_out,
    const int kernel_size,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group
) {
    // Each block computes one output element
    int output_index = blockIdx.x;
    int total_outputs = batch_size * out_channels * h_out * w_out;
    if (output_index >= total_outputs) return;

    // Decode output index into (n, c, h, w)
    int w = output_index % w_out;
    int tmp = output_index / w_out;
    int h = tmp % h_out;
    tmp = tmp / h_out;
    int c = tmp % out_channels;
    int n = tmp / out_channels;

    int g = c / out_channels_per_group; // group index
    int c_local = c % out_channels_per_group;
    
    // Each thread computes a partial sum over the reduction of kernel window and in_channels_per_group dimension
    float partial = 0.0f;

    // Loop over the kernel height and width
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in_candidate = h + padding_h - kh;
        if (h_in_candidate % stride_h != 0) continue;
        int h_in_idx = h_in_candidate / stride_h;
        if (h_in_idx < 0 || h_in_idx >= h_in) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in_candidate = w + padding_w - kw;
            if (w_in_candidate % stride_w != 0) continue;
            int w_in_idx = w_in_candidate / stride_w;
            if (w_in_idx < 0 || w_in_idx >= w_in) continue;
            // Reduce over the in_channels per group dimension in a strided manner across threads
            for (int r = threadIdx.x; r < in_channels_per_group; r += blockDim.x) {
                int in_channel = g * in_channels_per_group + r;
                int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                partial += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    // Intra-block reduction using shared memory and warp-level primitives
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = partial;
    __syncthreads();

    // Standard reduction in shared memory until 32 threads remain
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Use warp-level reduction for the final stage
    if (threadIdx.x < 32) {
        volatile float* smem = sdata;
        // Unroll warp-level reduction using __shfl_down_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            float val = __shfl_down_sync(0xffffffff, smem[threadIdx.x], offset);
            smem[threadIdx.x] += val;
        }
    }

    // Thread 0 writes the result
    if (threadIdx.x == 0) {
        float sum = sdata[0];
        if (bias != nullptr) {
            sum += __ldg(&bias[c]);
        }
        output[output_index] = sum;
    }
}

// Forward function that parses arguments, calculates dimensions, and launches the kernel
// Each block computes one output element; block-level reduction uses shared memory of size (blockDim.x * sizeof(float))

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

    int stride_h = stride_vec[0];
    int stride_w = (stride_vec.size() > 1) ? stride_vec[1] : stride_h;
    int padding_h = padding_vec[0];
    int padding_w = (padding_vec.size() > 1) ? padding_vec[1] : padding_h;
    int output_padding_h = output_padding_vec[0];
    int output_padding_w = (output_padding_vec.size() > 1) ? output_padding_vec[1] : output_padding_h;

    // Input dimensions: [batch_size, in_channels, h_in, w_in]
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int h_in = x.size(2);
    const int w_in = x.size(3);

    // Weight dimensions: [in_channels, out_channels_per_group, kernel_size, kernel_size]
    const int kernel_size = weight.size(2); // assume square kernel
    int out_channels = weight.size(1) * groups;

    // Calculate output dimensions for transposed convolution
    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output_tensor = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    // Total number of output elements
    int total_outputs = batch_size * out_channels * h_out * w_out;
    // Launch one block per output element
    int block_size = 256; // Choose an appropriate block size for reduction
    int grid_size = total_outputs;
    
    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);
    float* output_ptr = output_tensor.data_ptr<float>();

    // Launch kernel with dynamic shared memory size = block_size * sizeof(float)
    conv_transpose2d_shared_reduce_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        h_in,
        w_in,
        out_channels,
        h_out,
        w_out,
        kernel_size,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        groups,
        in_channels_per_group,
        out_channels_per_group
    );

    cudaDeviceSynchronize();
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward kernel with shared memory reduction and warp-level primitives",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
