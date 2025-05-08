#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_aligned_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    bool return_indices)
{
    const int elements_per_bc = output_length;
    const int total_elements = batch_size * num_channels * output_length;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_elements) return;
    
    const int bc = tid / elements_per_bc;
    const int i = tid % elements_per_bc;
    const int b = bc / num_channels;
    const int c = bc % num_channels;
    
    if (b >= batch_size || c >= num_channels) return;

    const int input_start = i * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    // Base input address for current batch and channel
    const float* input_ptr = input + b * num_channels * input_length + c * input_length;

    // Align kernel iterations to process 4 elements at once where possible
    const int aligned_start = (input_start + 3) & ~3;
    const int aligned_end = (input_start + kernel_size * dilation) & ~3;

    // Handle pre-alignment elements
    for (int k = input_start; k < aligned_start && k < input_start + kernel_size * dilation; k += dilation) {
        if (k >= 0 && k < input_length) {
            const float val = __ldg(input_ptr + k);
            if (val > max_val) {
                max_val = val;
                max_idx = k;
            }
        }
    }

    // Process aligned elements using float4
    for (int k = aligned_start; k < aligned_end; k += 4 * dilation) {
        if (k >= 0 && k + 3 * dilation < input_length) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int pos = k + j * dilation;
                if (pos < input_start + kernel_size * dilation) {
                    const float val = __ldg(input_ptr + pos);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = pos;
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (int k = aligned_end; k < input_start + kernel_size * dilation; k += dilation) {
        if (k >= 0 && k < input_length) {
            const float val = __ldg(input_ptr + k);
            if (val > max_val) {
                max_val = val;
                max_idx = k;
            }
        }
    }

    const int out_idx = b * num_channels * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (return_indices) indices[out_idx] = max_idx;
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices)
{
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;

    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
            options.dtype(torch::kInt64));
    }

    const int total_elements = batch_size * num_channels * output_length;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    max_pool1d_aligned_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with aligned memory access (CUDA)");
}