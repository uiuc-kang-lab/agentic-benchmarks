#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Each warp computes one output element using warp-level reduction.

template <typename scalar_t>
__global__ void warp_max_pool3d_forward_kernel(
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

    // Each warp (32 threads) computes one output element
    const int warpSize = 32;
    int warpId = (blockIdx.x * (blockDim.x / warpSize)) + (threadIdx.x / warpSize);
    int lane = threadIdx.x % warpSize;

    int total_output = batch_size * channels * output_d * output_h * output_w;
    if (warpId >= total_output) return;

    // Decode output element index in the order: [b, c, d, h, w]
    int idx = warpId;
    int w_out = idx % output_w;
    idx /= output_w;
    int h_out = idx % output_h;
    idx /= output_h;
    int d_out = idx % output_d;
    idx /= output_d;
    int c = idx % channels;
    idx /= channels;
    int b = idx;

    // Compute the start indices in the input
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Total number of elements in the pooling window
    int pool_size = kernel_size * kernel_size * kernel_size;
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
    int local_idx = -1;

    // Each lane handles a subset of the pooling window entries
    for (int i = lane; i < pool_size; i += warpSize) {
        int k_d = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int k_h = rem / kernel_size;
        int k_w = rem % kernel_size;

        int d_in = d_start + k_d * dilation;
        int h_in = h_start + k_h * dilation;
        int w_in = w_start + k_w * dilation;

        if (d_in < 0 || d_in >= input_d || h_in < 0 || h_in >= input_h || w_in < 0 || w_in >= input_w) {
            continue;
        }

        int input_idx = ((b * channels + c) * input_d + d_in) * (input_h * input_w) + h_in * input_w + w_in;
        scalar_t val = input[input_idx];
        if (val > local_max) {
            local_max = val;
            local_idx = input_idx;
        }
    }

    // Warp-level reduction using shuffle operations
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Lane 0 writes the result to global memory
    if (lane == 0) {
        output[warpId] = local_max;
        if (indices != nullptr) {
            indices[warpId] = local_idx;
        }
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

    // Compute output dimensions
    float d_calc = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float h_calc = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float w_calc = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;

    const int output_d = ceil_mode ? std::ceil(d_calc) : std::floor(d_calc);
    const int output_h = ceil_mode ? std::ceil(h_calc) : std::floor(h_calc);
    const int output_w = ceil_mode ? std::ceil(w_calc) : std::floor(w_calc);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ? torch::empty({batch_size, channels, output_d, output_h, output_w}, 
                              input.options().dtype(torch::kLong)) : torch::Tensor();

    int total_output = batch_size * channels * output_d * output_h * output_w;
    const int warp_size = 32;
    const int threads = 256;
    int warps_per_block = threads / warp_size;
    int blocks = (total_output + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_max_pool3d_forward_cuda", ([&] {
        warp_max_pool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward (CUDA)");
}
