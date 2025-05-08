#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel_optimized(
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
    
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int d_out = blockIdx.z * blockDim.z + threadIdx.z;
    if (w_out >= output_w || h_out >= output_h || d_out >= output_d) return;

    const int c = blockIdx.w;
    const int b = blockIdx.u;

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    for (int k_d = 0; k_d < kernel_size; k_d++) {
        const int d_in = d_start + k_d * dilation;
        if (d_in < 0 || d_in >= input_d) continue;

        for (int k_h = 0; k_h < kernel_size; k_h++) {
            const int h_in = h_start + k_h * dilation;
            if (h_in < 0 || h_in >= input_h) continue;

            for (int k_w = 0; k_w < kernel_size; k_w++) {
                const int w_in = w_start + k_w * dilation;
                if (w_in < 0 || w_in >= input_w) continue;

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

    const int output_idx = ((b * channels + c) * output_d + d_out) * output_h * output_w + h_out * output_w + w_out;
    output[output_idx] = max_val;
    if (indices != nullptr) {
        indices[output_idx] = max_index;
    }
}

torch::Tensor max_pool3d_cuda_forward_optimized(
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

    const dim3 threads(8, 8, 8);
    const dim3 blocks((output_w + threads.x - 1) / threads.x, (output_h + threads.y - 1) / threads.y, (output_d + threads.z - 1) / threads.z, channels, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_forward_cuda_optimized", ([&] {
        max_pool3d_forward_kernel_optimized<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool3d_cuda_forward_optimized, "Max Pool 3D forward optimized (CUDA)");
}