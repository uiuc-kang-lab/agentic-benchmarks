#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel performs 1D max pooling without using shared memory,
// hence it avoids any __syncthreads() since each thread loads its own data
// directly from global memory using __ldg() for efficient read-only caching.
// This minimizes synchronization overhead while ensuring correctness.

__global__ void max_pool1d_no_sync_kernel(
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
    bool return_indices) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_channels * output_length;

    if (idx < total) {
        // Decode the flattened index into batch, channel, and output position
        int o = idx % output_length;
        int nc = idx / output_length;
        int c = nc % num_channels;
        int b = nc / num_channels;

        // Compute the starting index in the input corresponding to this output element
        int in_start = o * stride - padding;
        float max_val = -INFINITY;
        int max_idx = -1;

        int base = b * num_channels * input_length + c * input_length;

        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int pos = in_start + k * dilation;
            if (pos >= 0 && pos < input_length) {
                float val = __ldg(&input[base + pos]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = pos;
                }
            }
        }

        output[idx] = max_val;
        if (return_indices) {
            indices[idx] = max_idx;
        }
    }
}

// Host function wrapping the CUDA kernel launch

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    // Calculate output length based on the pooling parameters
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    int total = batch_size * num_channels * output_length;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    max_pool1d_no_sync_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "MaxPool1D forward without unnecessary __syncthreads (CUDA)");
}
