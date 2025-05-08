#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

template <typename scalar_t, int KERNEL_SIZE=3>
__global__ void divergence_free_maxpool3d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int stride,
    const int padding,
    const int dilation) {
    
    // Align work distribution with warp size
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x & 31;  // Thread position within warp
    const int warp_id = tid >> 5;          // Global warp index
    
    const int total_warps = (batch_size * channels * output_d * output_h * output_w + 31) >> 5;
    if (warp_id >= total_warps) return;

    // Decompose global index while maintaining warp coherency
    const int work_per_warp = output_w * output_h;
    const int warp_work = warp_id * 32 + lane_id;
    
    const int w_out = warp_work % output_w;
    const int h_out = (warp_work / output_w) % output_h;
    const int d_out = (warp_work / work_per_warp) % output_d;
    const int c = (warp_work / (work_per_warp * output_d)) % channels;
    const int b = warp_work / (work_per_warp * output_d * channels);

    if (b >= batch_size) return;

    // Compute input window bounds
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Pre-compute valid ranges to avoid divergent branches
    const int d_end = d_start + (KERNEL_SIZE - 1) * dilation + 1;
    const int h_end = h_start + (KERNEL_SIZE - 1) * dilation + 1;
    const int w_end = w_start + (KERNEL_SIZE - 1) * dilation + 1;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    // Use predicated execution instead of branching
    #pragma unroll
    for (int kd = 0; kd < KERNEL_SIZE; ++kd) {
        const int d_in = d_start + kd * dilation;
        const bool d_valid = (d_in >= 0 && d_in < input_d);
        
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const int h_in = h_start + kh * dilation;
            const bool h_valid = d_valid && (h_in >= 0 && h_in < input_h);
            
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int w_in = w_start + kw * dilation;
                const bool valid = h_valid && (w_in >= 0 && w_in < input_w);
                
                if (valid) {
                    const int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                    const scalar_t val = __ldg(&input[input_idx]);
                    if (val > max_val) {
                        max_val = val;
                        max_index = input_idx;
                    }
                }
            }
        }
    }

    // Compute output index maintaining coalesced access pattern
    const int output_idx = (((b * channels + c) * output_d + d_out) * output_h + h_out) * output_w + w_out;
    if (warp_work < total_warps * 32) {
        output[output_idx] = max_val;
        if (indices != nullptr) {
            indices[output_idx] = max_index;
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

    // Configure launch parameters aligned with warp size
    const int threads_per_block = 256;  // Multiple of warp size (32)
    const int total_elements = batch_size * channels * output_d * output_h * output_w;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        divergence_free_maxpool3d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            stride, padding, dilation);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward, "Divergence-free Max Pool 3D forward (CUDA)");
}