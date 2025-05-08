#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// This kernel uses a 3D grid mapping: threads in x and y cover the spatial output (w and h),
// while threads in z cover the combined dimension for (batch, channel, depth).

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel(
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

    // Map thread indices to output coordinates using a 3D grid
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;  // output width index
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;  // output height index
    int bcd_index = blockIdx.z * blockDim.z + threadIdx.z;  // combined index for batch, channel, and depth
    
    int total_bcd = batch_size * channels * output_d;
    if (w_out >= output_w || h_out >= output_h || bcd_index >= total_bcd) return;
    
    // Decode bcd_index into depth, channel, and batch indices
    int d_out = bcd_index % output_d;
    int tmp = bcd_index / output_d;
    int c = tmp % channels;
    int b = tmp / channels;

    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    #pragma unroll
    for (int k_d = 0; k_d < kernel_size; k_d++) {
        int d_in = d_start + k_d * dilation;
        if (d_in < 0 || d_in >= input_d) continue;
        
        #pragma unroll
        for (int k_h = 0; k_h < kernel_size; k_h++) {
            int h_in = h_start + k_h * dilation;
            if (h_in < 0 || h_in >= input_h) continue;
            
            #pragma unroll
            for (int k_w = 0; k_w < kernel_size; k_w++) {
                int w_in = w_start + k_w * dilation;
                if (w_in < 0 || w_in >= input_w) continue;
                
                int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                h_in * input_w + w_in;
                scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    int out_idx = ((b * channels + c) * output_d + d_out) * output_h * output_w +
                  h_out * output_w + w_out;
    output[out_idx] = max_val;
    if (indices != nullptr) {
        indices[out_idx] = max_index;
    }
}

// Wrapper function
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

    int total_bcd = batch_size * channels * output_d;
    // Define block and grid dimensions for the 3D mapping
    dim3 block(16, 16, 4);
    dim3 grid((output_w + block.x - 1) / block.x,
              (output_h + block.y - 1) / block.y,
              (total_bcd + block.z - 1) / block.z);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel<scalar_t><<<grid, block>>>(
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
