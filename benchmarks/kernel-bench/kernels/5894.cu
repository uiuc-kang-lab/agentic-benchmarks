#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Optimized kernel combining advantages of both approaches
// Using block-level shared memory for reduction and mid-level thread parallelism

template <typename scalar_t>
__global__ void optimized_max_pool3d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * channels * output_d * output_h * output_w) return;

    // Decode the linear output index
    const int w_out = out_idx % output_w;
    const int h_out = (out_idx / output_w) % output_h;
    const int d_out = (out_idx / (output_w * output_h)) % output_d;
    const int c = (out_idx / (output_w * output_h * output_d)) % channels;
    const int b = out_idx / (output_w * output_h * output_d * channels);

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    int64_t local_index = -1;

    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        const int d_in = d_start + k_d * dilation;
        if (d_in < 0 || d_in >= input_d) continue;

        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            const int h_in = h_start + k_h * dilation;
            if (h_in < 0 || h_in >= input_h) continue;

            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                const int w_in = w_start + k_w * dilation;
                if (w_in < 0 || w_in >= input_w) continue;

                const int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                      h_in * input_w + w_in;
                scalar_t val = input[input_idx];

                if (val > local_max) {
                    local_max = val;
                    local_index = input_idx;
                }
            }
        }
    }

    __shared__ scalar_t shared_max[128];  // Consider maximum threads per block used
    __shared__ int64_t shared_index[128];

    shared_max[threadIdx.x] = local_max;
    shared_index[threadIdx.x] = local_index;
    __syncthreads();

    // Reduction within a block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            if (shared_max[threadIdx.x + offset] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + offset];
                shared_index[threadIdx.x] = shared_index[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out_idx] = shared_max[0];
        if (indices != nullptr) {
            indices[out_idx] = shared_index[0];
        }
    }
}

torch::Tensor optimized_max_pool3d_cuda_forward(
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

    const int output_d = ceil_mode ? 
        static_cast<int>(ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)) :
        static_cast<int>(floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1));
    const int output_h = ceil_mode ?
        static_cast<int>(ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)) :
        static_cast<int>(floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1));
    const int output_w = ceil_mode ?
        static_cast<int>(ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)) :
        static_cast<int>(floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1));

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int total_outputs = batch_size * channels * output_d * output_h * output_w;
    const int threads = 128;
    const int blocks = (total_outputs + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_max_pool3d_forward_cuda", ([&] {
        optimized_max_pool3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size, stride, padding, dilation);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_max_pool3d_cuda_forward, "Optimized Max Pool 3D forward (CUDA)");
}