#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_kernel_shared(
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
    extern __shared__ float shared_data[];
    float* shared_vals = shared_data;
    int* shared_idx = (int*)&shared_vals[blockDim.x];

    const int b = blockIdx.z;
    const int c = blockIdx.y;
    const int i = blockIdx.x;  // each block computes one output element

    if (b >= batch_size || c >= num_channels || i >= output_length) return;

    const int input_start = i * stride - padding;
    float local_max = -INFINITY;
    int local_idx = -1;

    // Each thread loads a portion of the pooling window
    for (int k = threadIdx.x; k < kernel_size; k += blockDim.x) {
        const int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            const float val = input[b * num_channels * input_length + c * input_length + pos];
            if (val > local_max) {
                local_max = val;
                local_idx = pos;
            }
        }
    }

    shared_vals[tid] = max_val;
    shared_idx[tid] = max_idx;
    __syncthreads();

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    // Write results
    if (i < output_length) {
        const int out_idx = b * num_channels * output_length + c * output_length + i;
        output[out_idx] = max_val;
        if (return_indices) {
            indices[out_idx] = max_idx;
        }
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
            options.dtype(torch::kInt64));
    }

    const int threads_per_block = 256;
    const dim3 threads(threads_per_block);
    const dim3 blocks(
        (output_length + threads_per_block - 1) / threads_per_block,
        num_channels,
        batch_size
    );

    const size_t shared_mem_size = threads_per_block * (sizeof(float) + sizeof(int));

    max_pool1d_kernel_shared<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with shared memory reduction (CUDA)");
}