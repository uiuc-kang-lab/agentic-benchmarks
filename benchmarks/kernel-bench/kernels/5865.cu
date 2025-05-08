#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

// Structure to hold pooling parameters in constant memory
struct PoolParams {
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int input_d;
    int input_h;
    int input_w;
    int output_d;
    int output_h;
    int output_w;
};

// Declare the constant memory variable
__constant__ PoolParams cPoolParams;

// CUDA kernel that uses constant memory for pooling parameters
// Each thread computes one output element of the 3D max pooling operation

template <typename scalar_t>
__global__ void max_pool3d_forward_const_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels) {

    int total = batch_size * channels * cPoolParams.output_d * cPoolParams.output_h * cPoolParams.output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode output indices using constant memory values
    int w_out = idx % cPoolParams.output_w;
    int h_out = (idx / cPoolParams.output_w) % cPoolParams.output_h;
    int d_out = (idx / (cPoolParams.output_w * cPoolParams.output_h)) % cPoolParams.output_d;
    int tmp = idx / (cPoolParams.output_w * cPoolParams.output_h * cPoolParams.output_d);
    int c = tmp % channels;
    int b = tmp / channels;

    // Compute the starting index in the input tensor for the pooling window
    int d_start = d_out * cPoolParams.stride - cPoolParams.padding;
    int h_start = h_out * cPoolParams.stride - cPoolParams.padding;
    int w_start = w_out * cPoolParams.stride - cPoolParams.padding;

    // Compute valid bounds for the pooling window to avoid out-of-bound accesses
    int k_d_min = (d_start < 0) ? ((-d_start + cPoolParams.dilation - 1) / cPoolParams.dilation) : 0;
    int valid_d_max = (cPoolParams.input_d - d_start + cPoolParams.dilation - 1) / cPoolParams.dilation;
    int k_d_max = (cPoolParams.kernel_size < valid_d_max) ? cPoolParams.kernel_size : valid_d_max;

    int k_h_min = (h_start < 0) ? ((-h_start + cPoolParams.dilation - 1) / cPoolParams.dilation) : 0;
    int valid_h_max = (cPoolParams.input_h - h_start + cPoolParams.dilation - 1) / cPoolParams.dilation;
    int k_h_max = (cPoolParams.kernel_size < valid_h_max) ? cPoolParams.kernel_size : valid_h_max;

    int k_w_min = (w_start < 0) ? ((-w_start + cPoolParams.dilation - 1) / cPoolParams.dilation) : 0;
    int valid_w_max = (cPoolParams.input_w - w_start + cPoolParams.dilation - 1) / cPoolParams.dilation;
    int k_w_max = (cPoolParams.kernel_size < valid_w_max) ? cPoolParams.kernel_size : valid_w_max;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    // Loop over the valid pooling window with loop unrolling to reduce overhead
    #pragma unroll
    for (int kd = k_d_min; kd < k_d_max; kd++) {
        int d_in = d_start + kd * cPoolParams.dilation;
        #pragma unroll
        for (int kh = k_h_min; kh < k_h_max; kh++) {
            int h_in = h_start + kh * cPoolParams.dilation;
            #pragma unroll
            for (int kw = k_w_min; kw < k_w_max; kw++) {
                int w_in = w_start + kw * cPoolParams.dilation;
                int input_idx = (((b * channels + c) * cPoolParams.input_d + d_in) * cPoolParams.input_h + h_in) * cPoolParams.input_w + w_in;
                scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    int out_idx = (((b * channels + c) * cPoolParams.output_d + d_out) * cPoolParams.output_h + h_out) * cPoolParams.output_w + w_out;
    output[out_idx] = max_val;
    if (indices != nullptr) {
        indices[out_idx] = max_index;
    }
}

// Host function to prepare constant memory and launch the CUDA kernel

torch::Tensor max_pool3d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];

    // Compute output dimensions
    float d_out_f = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float h_out_f = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float w_out_f = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;

    int output_d = ceil_mode ? std::ceil(d_out_f) : std::floor(d_out_f);
    int output_h = ceil_mode ? std::ceil(h_out_f) : std::floor(h_out_f);
    int output_w = ceil_mode ? std::ceil(w_out_f) : std::floor(w_out_f);

    // Set up constant memory with pooling parameters
    PoolParams hPoolParams;
    hPoolParams.kernel_size = kernel_size;
    hPoolParams.stride = stride;
    hPoolParams.padding = padding;
    hPoolParams.dilation = dilation;
    hPoolParams.input_d = input_d;
    hPoolParams.input_h = input_h;
    hPoolParams.input_w = input_w;
    hPoolParams.output_d = output_d;
    hPoolParams.output_h = output_h;
    hPoolParams.output_w = output_w;
    cudaMemcpyToSymbol(cPoolParams, &hPoolParams, sizeof(PoolParams));

    // Create output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    torch::Tensor indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    int total = batch_size * channels * output_d * output_h * output_w;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_const", ([&] {
        max_pool3d_forward_const_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            channels);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward with constant memory optimization (CUDA)");
}
