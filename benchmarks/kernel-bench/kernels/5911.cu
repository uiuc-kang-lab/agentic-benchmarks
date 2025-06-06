#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t* indices,
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

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;
    const int lane_id = threadIdx.x % 32;
    
    // Divide kernel window into warp-sized chunks
    for (int k_d = lane_id; k_d < kernel_size; k_d += 32) {
        const int d_in = d_start + k_d * dilation;
        if (d_in >= 0 && d_in < input_d) {
            for (int k_h = 0; k_h < kernel_size; k_h++) {
                const int h_in = h_start + k_h * dilation;
                if (h_in >= 0 && h_in < input_h) {
                    for (int k_w = 0; k_w < kernel_size; k_w++) {
                        const int w_in = w_start + k_w * dilation;
                        if (w_in >= 0 && w_in < input_w) {
                            const int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                                h_in * input_w + w_in;
                            const scalar_t val = input[input_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_index = input_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce maximum within warp
    scalar_t warp_max = warp_reduce_max(max_val);
    
    // First thread in warp writes result
    if (lane_id == 0) {
        output[idx] = warp_max;
        if (indices != nullptr) {
            indices[idx] = max_index;
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

    const int output_d = ceil_mode ? 
        ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_h = ceil_mode ?
        ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_w = ceil_mode ?
        ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ? 
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int threads = 256;
    const int blocks = (batch_size * channels * output_d * output_h * output_w + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
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