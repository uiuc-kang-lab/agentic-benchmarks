#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Constant memory for kernel parameters
__constant__ int c_kernel_size;
__constant__ int c_stride;
__constant__ int c_padding;
__constant__ int c_dilation;
__constant__ int c_input_dims[3];  // d, h, w
__constant__ int c_output_dims[3]; // d, h, w

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel_constant(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * c_output_dims[0] * c_output_dims[1] * c_output_dims[2];
    if (idx >= total) return;

    // Compute output indices
    const int w_out = idx % c_output_dims[2];
    const int h_out = (idx / c_output_dims[2]) % c_output_dims[1];
    const int d_out = (idx / (c_output_dims[2] * c_output_dims[1])) % c_output_dims[0];
    const int c = (idx / (c_output_dims[2] * c_output_dims[1] * c_output_dims[0])) % channels;
    const int b = idx / (c_output_dims[2] * c_output_dims[1] * c_output_dims[0] * channels);

    // Compute input window bounds
    const int d_start = d_out * c_stride - c_padding;
    const int h_start = h_out * c_stride - c_padding;
    const int w_start = w_out * c_stride - c_padding;

    // Compute valid kernel bounds
    const int k_d_start = (d_start < 0) ? ((-d_start + c_dilation - 1) / c_dilation) : 0;
    const int k_d_end = min(c_kernel_size, (c_input_dims[0] - d_start + c_dilation - 1) / c_dilation);

    const int k_h_start = (h_start < 0) ? ((-h_start + c_dilation - 1) / c_dilation) : 0;
    const int k_h_end = min(c_kernel_size, (c_input_dims[1] - h_start + c_dilation - 1) / c_dilation);

    const int k_w_start = (w_start < 0) ? ((-w_start + c_dilation - 1) / c_dilation) : 0;
    const int k_w_end = min(c_kernel_size, (c_input_dims[2] - w_start + c_dilation - 1) / c_dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    #pragma unroll
    for (int k_d = k_d_start; k_d < k_d_end; k_d++) {
        const int d_in = d_start + k_d * c_dilation;
        #pragma unroll
        for (int k_h = k_h_start; k_h < k_h_end; k_h++) {
            const int h_in = h_start + k_h * c_dilation;
            #pragma unroll
            for (int k_w = k_w_start; k_w < k_w_end; k_w++) {
                const int w_in = w_start + k_w * c_dilation;
                const int input_idx = (((b * channels + c) * c_input_dims[0] + d_in) * c_input_dims[1] + h_in) * c_input_dims[2] + w_in;
                const scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}

torch::Tensor max_pool3d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    // Calculate output dimensions
    const int output_d = ceil_mode ?
        ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_h = ceil_mode ?
        ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_w = ceil_mode ?
        ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    // Copy constant parameters to device constant memory
    int input_dims[3] = {input_d, input_h, input_w};
    int output_dims[3] = {output_d, output_h, output_w};
    cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(c_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(c_dilation, &dilation, sizeof(int));
    cudaMemcpyToSymbol(c_input_dims, input_dims, 3 * sizeof(int));
    cudaMemcpyToSymbol(c_output_dims, output_dims, 3 * sizeof(int));

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int threads = 256;
    const int total = batch_size * channels * output_d * output_h * output_w;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel_constant<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward with constant memory (CUDA)");
}