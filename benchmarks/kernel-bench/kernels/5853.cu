#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Kernel that maps the 5D output tensor (batch, channel, depth, height, width) onto a 3D grid.
// The grid dimensions: 
//   grid.x: covers output width
//   grid.y: covers output height
//   grid.z: covers batch * channels * output_depth
// Each thread computes one output element with coordinates (b, c, d, h, w).

template <typename scalar_t>
__global__ void max_pool3d_even_workload_kernel(
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

    // Compute output spatial coordinates for width and height
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Flatten batch, channel, and output depth into the z-dimension
    int index_z = blockIdx.z * blockDim.z + threadIdx.z;
    int total_z = batch_size * channels * output_d;

    if (w_out < output_w && h_out < output_h && index_z < total_z) {
        // Decode index_z into output depth, channel and batch
        int d_out = index_z % output_d;
        int temp = index_z / output_d;
        int c = temp % channels;
        int b = temp / channels;

        // Compute the starting indices in the input tensor
        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        // Calculate valid pooling window boundaries to avoid out-of-bound accesses
        int k_d_start = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
        int k_d_end = min(kernel_size, (input_d - d_start + dilation - 1) / dilation);
        int k_h_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
        int k_h_end = min(kernel_size, (input_h - h_start + dilation - 1) / dilation);
        int k_w_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
        int k_w_end = min(kernel_size, (input_w - w_start + dilation - 1) / dilation);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int max_index = -1;

        // Iterate over the valid pooling window
        for (int kd = k_d_start; kd < k_d_end; kd++) {
            int d_in = d_start + kd * dilation;
            for (int kh = k_h_start; kh < k_h_end; kh++) {
                int h_in = h_start + kh * dilation;
                for (int kw = k_w_start; kw < k_w_end; kw++) {
                    int w_in = w_start + kw * dilation;
                    int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                    scalar_t val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_index = input_idx;
                    }
                }
            }
        }

        // Compute the output index in the 5D tensor flattened in row-major order
        int output_idx = ((((b * channels + c) * output_d + d_out) * output_h + h_out) * output_w) + w_out;
        output[output_idx] = max_val;
        if (indices != nullptr) {
            indices[output_idx] = max_index;
        }
    }
}

// Host function: sets up kernel launch parameters using a 3D grid and 3D block to achieve even workload distribution

torch::Tensor max_pool3d_cuda_forward_even(
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
    float d_out_f = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1.0f;
    float h_out_f = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1.0f;
    float w_out_f = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1.0f;
    const int output_d = ceil_mode ? std::ceil(d_out_f) : std::floor(d_out_f);
    const int output_h = ceil_mode ? std::ceil(h_out_f) : std::floor(h_out_f);
    const int output_w = ceil_mode ? std::ceil(w_out_f) : std::floor(w_out_f);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    // Set up 3D block and grid dimensions
    const dim3 block(8, 8, 4);
    const dim3 grid(
        (output_w + block.x - 1) / block.x,
        (output_h + block.y - 1) / block.y,
        ((batch_size * channels * output_d) + block.z - 1) / block.z);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_even", ([&] {
        max_pool3d_even_workload_kernel<scalar_t><<<grid, block>>>(
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
    m.def("forward", &max_pool3d_cuda_forward_even, "Max Pool 3D forward even workload (CUDA)");
}
