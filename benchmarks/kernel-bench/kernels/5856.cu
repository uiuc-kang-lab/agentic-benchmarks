#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Optimized 3D Max Pooling Kernel
// This kernel refines the thread and block indexing to efficiently map threads to the 3D pooling grid.

const int MAX_THREADS = 512; // Upper limit for sanity check/ceiling for threads per block

// Compute the start position in input tensor (w, h, d) from output indices
__device__ inline int compute_start(int out_idx, int stride, int padding) {
    return out_idx * stride - padding;
}

// Compute the input index from spatial (w, h, d) and channel/batch positions
__device__ inline int compute_input_index(
    int b, int c, int d, int h, int w,
    int channels, int input_d, int input_h, int input_w) {
    return (((b * channels + c) * input_d + d) * input_h + h) * input_w + w;
}

// Combined kernel with optimized mapping from block/thread indices to the problem domain
// Utilizing 3D grid and 3D block configuration for better exploration of the problem space

template <typename scalar_t>
__global__ void max_pool3d_forward_indexing_kernel(
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
    int d_out = blockIdx.z * blockDim.z + threadIdx.z;

    if (w_out >= output_w || h_out >= output_h || d_out >= output_d) return;

    int idx = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) 
              * blockDim.x * blockDim.y * blockDim.z
              + (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    int c = (idx / (output_w * output_h * output_d)) % channels;
    int b = idx / (output_w * output_h * output_d * channels);  

    // Compute input tensor bounds
    int d_start = compute_start(d_out, stride, padding);
    int h_start = compute_start(h_out, stride, padding);
    int w_start = compute_start(w_out, stride, padding);
    
    int k_d_start = d_start < 0 ? (-d_start + dilation - 1) / dilation : 0;
    int k_d_end = min(kernel_size, (input_d - d_start + dilation - 1) / dilation);

    int k_h_start = h_start < 0 ? (-h_start + dilation - 1) / dilation : 0;
    int k_h_end = min(kernel_size, (input_h - h_start + dilation - 1) / dilation);

    int k_w_start = w_start < 0 ? (-w_start + dilation - 1) / dilation : 0;
    int k_w_end = min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

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
                int input_index = compute_input_index(b, c, d_in, h_in, w_in,
                    channels, input_d, input_h, input_w);
                scalar_t val = input[input_index];
                if (val > max_val) {
                    max_val = val;
                    max_index = input_index;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}

// Host function to setup and launch the kernel

torch::Tensor max_pool3d_cuda_forward_indexing(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    const auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    // Calculate output dimensions
    int output_d = ceil_mode ? ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
                             : floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    int output_h = ceil_mode ? ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
                             : floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    int output_w = ceil_mode ? ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
                             : floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ? 
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    dim3 block_dim(16, 16, 4); // Configured to optimize performance balancing dimensions
    dim3 grid_dim(
        (output_w + block_dim.x - 1) / block_dim.x,
        (output_h + block_dim.y - 1) / block_dim.y,
        (output_d + block_dim.z - 1) / block_dim.z);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_indexing", ([&] {
        max_pool3d_forward_indexing_kernel<scalar_t><<<grid_dim, block_dim>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            channels,
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
    m.def("forward", &max_pool3d_cuda_forward_indexing, "Optimized Max Pool 3D forward with indexing (CUDA)");
}