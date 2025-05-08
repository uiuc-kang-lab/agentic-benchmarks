#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

// This kernel uses atomic operations to handle race conditions when multiple threads
// might write to the same output location. Atomic operations are minimized to reduce
// contention and are only used when necessary.

// Atomic max function for floating point numbers
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Optimized CUDA kernel for 3D max pooling using atomic operations
// to handle potential race conditions in global memory

template <typename scalar_t>
__global__ void atomic_maxpool3d_kernel(
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

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_idx = blockIdx.z;

    int d_out = linear_idx % output_d;
    int tmp = linear_idx / output_d;
    int c = tmp % channels;
    int b = tmp / channels;

    if (w_out >= output_w || h_out >= output_h) return;

    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    int k_d_start = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    int k_d_end = std::min(kernel_size, (input_d - d_start + dilation - 1) / dilation);

    int k_h_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    int k_h_end = std::min(kernel_size, (input_h - h_start + dilation - 1) / dilation);

    int k_w_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    int k_w_end = std::min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    #pragma unroll
    for (int kd = k_d_start; kd < k_d_end; kd++) {
        int d_in = d_start + kd * dilation;
        #pragma unroll
        for (int kh = k_h_start; kh < k_h_end; kh++) {
            int h_in = h_start + kh * dilation;
            #pragma unroll
            for (int kw = k_w_start; kw < k_w_end; kw++) {
                int w_in = w_start + kw * dilation;
                int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                scalar_t val = __ldg(&input[input_idx]);
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    int output_idx = (((b * channels + c) * output_d + d_out) * output_h + h_out) * output_w + w_out;
    atomicMaxFloat(&output[output_idx], max_val);
    if (indices != nullptr && max_val == output[output_idx]) {
        indices[output_idx] = max_index;
    }
}

// Host function to launch the optimized kernel with atomic operations

torch::Tensor max_pool3d_cuda_forward_atomic(
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

    float d_out_f = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float h_out_f = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;
    float w_out_f = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1;

    int output_d = ceil_mode ? std::ceil(d_out_f) : std::floor(d_out_f);
    int output_h = ceil_mode ? std::ceil(h_out_f) : std::floor(h_out_f);
    int output_w = ceil_mode ? std::ceil(w_out_f) : std::floor(w_out_f);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    torch::Tensor indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    dim3 block(32, 8);
    dim3 grid(
        (output_w + block.x - 1) / block.x,
        (output_h + block.y - 1) / block.y,
        batch_size * channels * output_d
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_atomic", ([&] {
        atomic_maxpool3d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size, stride, padding, dilation
        );
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward_atomic, "Atomic Max Pool 3D forward (CUDA)");
}
