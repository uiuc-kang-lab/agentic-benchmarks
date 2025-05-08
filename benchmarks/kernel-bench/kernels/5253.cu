#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_kernel_optimized(
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
    const bool return_indices) 
{
    extern __shared__ float shared_input[];
    
    const int tid = threadIdx.x;
    const int b = blockIdx.z;
    const int c = blockIdx.y;
    const int out_idx = tid + blockIdx.x * blockDim.x;
    
    if (out_idx >= output_length) return;

    const int input_start = out_idx * stride - padding;
    const int shared_mem_size = (blockDim.x * stride + kernel_size * dilation);
    
    const int base_idx = b * num_channels * input_length + c * input_length;
    for (int i = tid; i < shared_mem_size; i += blockDim.x) {
        const int pos = input_start + i;
        shared_input[i] = (pos >= 0 && pos < input_length) ? 
            input[base_idx + pos] : -INFINITY;
    }
    __syncthreads();

    float max_val = -INFINITY;
    int max_idx = -1;
    
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        const int pos = tid * stride + k * dilation;
        const float val = shared_input[pos];
        if (val > max_val) {
            max_val = val;
            max_idx = input_start + k * dilation;
        }
    }

    const int out_offset = b * num_channels * output_length + c * output_length + out_idx;
    output[out_offset] = max_val;
    if (return_indices) {
        indices[out_offset] = max_idx;
    }
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
                             torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    }

    const int threads = 256;
    const dim3 blocks((output_length + threads - 1) / threads, num_channels, batch_size);
    const int shared_mem_size = (threads * stride + kernel_size * dilation) * sizeof(float);

    max_pool1d_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "MaxPool1D forward optimized (CUDA)");
}