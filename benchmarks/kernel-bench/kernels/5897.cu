#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Optimized kernel combining features of both kernels
// Uses a tunable block size and shared memory reduction for efficiency

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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c = (idx / (output_w * output_h * output_d)) % channels;
    const int b = idx / (output_w * output_h * output_d * channels);

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Allocate shared memory
    extern __shared__ char shared_mem[];
    scalar_t* sdata_val = reinterpret_cast<scalar_t*>(shared_mem);
    int64_t* sdata_idx = reinterpret_cast<int64_t*>(&sdata_val[blockDim.x]);

    scalar_t local_max = -std::numeric_limits<scalar_t>::max();
    int64_t local_index = 0;
    bool initialized = false;

    for (int i = 0; i < kernel_size * kernel_size * kernel_size; i++) {
        int k_d = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int k_h = rem / kernel_size;
        int k_w = rem % kernel_size;

        int d_in = d_start + k_d * dilation;
        int h_in = h_start + k_h * dilation;
        int w_in = w_start + k_w * dilation;

        if (d_in >= 0 && d_in < input_d &&
            h_in >= 0 && h_in < input_h &&
            w_in >= 0 && w_in < input_w) {
            int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                            h_in * input_w + w_in;
            scalar_t val = input[input_idx];
            if (val > local_max) {
                local_max = val;
                local_index = input_idx;
            }
        }
    }

    sdata_val[threadIdx.x] = local_max;
    sdata_idx[threadIdx.x] = local_index;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_val[threadIdx.x + s] > sdata_val[threadIdx.x]) {
                sdata_val[threadIdx.x] = sdata_val[threadIdx.x + s];
                sdata_idx[threadIdx.x] = sdata_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (blockDim.x > 32 && threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(0xffffffff, sdata_val[threadIdx.x], offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, sdata_idx[threadIdx.x], offset);
            if (other_val > sdata_val[threadIdx.x]) {
                sdata_val[threadIdx.x] = other_val;
                sdata_idx[threadIdx.x] = other_idx;
            }
        }
    }

    if (threadIdx.x == 0) {
        output[idx] = sdata_val[0];
        if (indices != nullptr) {
            indices[idx] = sdata_idx[0];
        }
    }
}

// Host wrapper function for the optimized max pooling kernel

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

    const int threads = 128; // Tuned for optimized performance
    const int blocks = (batch_size * channels * output_d * output_h * output_w + threads - 1) / threads;

    size_t shared_mem_size = threads * (sizeof(float) + sizeof(int64_t));
    if (input.scalar_type() == torch::kDouble) {
        shared_mem_size = threads * (sizeof(double) + sizeof(int64_t));
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_max_pool3d_forward_cuda", ([&] {
        optimized_max_pool3d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
