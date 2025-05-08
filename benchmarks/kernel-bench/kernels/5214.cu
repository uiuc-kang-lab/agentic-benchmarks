#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Structure to hold pooling parameters in constant memory
struct PoolParams {
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int input_length;
    int output_length;
};

// Allocate constant memory for pooling parameters
__constant__ PoolParams d_poolparams;

// CUDA kernel using constant memory for read-only pooling parameters
__global__ void max_pool1d_const_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels) {

    int total = batch_size * num_channels * d_poolparams.output_length;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop over output elements in a grid-stride loop
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        // Decode flat index into batch, channel, and output position indices
        int o = idx % d_poolparams.output_length;
        int c = (idx / d_poolparams.output_length) % num_channels;
        int b = idx / (d_poolparams.output_length * num_channels);

        int input_start = o * d_poolparams.stride - d_poolparams.padding;
        float max_val = -INFINITY;
        int max_idx = -1;

        int base_idx = b * num_channels * d_poolparams.input_length + c * d_poolparams.input_length;
        
        #pragma unroll
        for (int k = 0; k < d_poolparams.kernel_size; ++k) {
            int pos = input_start + k * d_poolparams.dilation;
            if (pos >= 0 && pos < d_poolparams.input_length) {
                float val = input[base_idx + pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = pos;
                }
            }
        }

        int out_idx = b * num_channels * d_poolparams.output_length + c * d_poolparams.output_length + o;
        output[out_idx] = max_val;
        if (indices) {
            indices[out_idx] = max_idx;
        }
    }
}

// Host function: sets up constant memory and launches the CUDA kernel
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

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    // Prepare pooling parameters and copy them to constant memory
    PoolParams h_poolparams;
    h_poolparams.kernel_size = kernel_size;
    h_poolparams.stride = stride;
    h_poolparams.padding = padding;
    h_poolparams.dilation = dilation;
    h_poolparams.input_length = input_length;
    h_poolparams.output_length = output_length;

    cudaMemcpyToSymbol(d_poolparams, &h_poolparams, sizeof(PoolParams), 0, cudaMemcpyHostToDevice);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
                                 options.dtype(torch::kInt64));
    }

    int total_elements = batch_size * num_channels * output_length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    max_pool1d_const_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with constant memory (CUDA)");
}
