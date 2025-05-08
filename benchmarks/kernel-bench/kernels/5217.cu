#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized device function using shared memory for max pooling
template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
__device__ inline void shared_max_pooling(
    const float* __restrict__ input,
    const int b,
    const int c,
    const int input_length,
    const int num_channels,
    const int input_start,
    const int kernel_size,
    const int dilation,
    float& max_val,
    int& max_idx,
    float* shared_data
) {
    const int tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    const int base = b * num_channels * input_length + c * input_length;
    
    // Initialize shared memory
    shared_data[tid] = -INFINITY;
    int local_max_idx = -1;
    
    // Load and process data in chunks
    #pragma unroll
    for (int k = 0; k < kernel_size; k += BLOCK_DIM_X * BLOCK_DIM_Y) {
        const int pos = input_start + (k + tid) * dilation;
        if (k + tid < kernel_size && pos >= 0 && pos < input_length) {
            float val = input[base + pos];
            if (val > shared_data[tid]) {
                shared_data[tid] = val;
                local_max_idx = pos;
            }
        }
    }
    
    __syncthreads();
    
    // Reduce within warp
    #pragma unroll
    for (int offset = (BLOCK_DIM_X * BLOCK_DIM_Y) / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shared_data[tid + offset] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + offset];
                local_max_idx = max_idx;
            }
        }
        __syncthreads();
    }
    
    max_val = shared_data[0];
    max_idx = local_max_idx;
}

__global__ void optimized_max_pool1d_kernel(
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
    const bool return_indices
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    
    if (b >= batch_size || c >= num_channels || i >= output_length) return;
    
    // Shared memory declaration
    __shared__ float shared_data[32 * 4];
    
    const int input_start = i * stride - padding;
    float max_val;
    int max_idx;
    
    shared_max_pooling<32, 4>(
        input, b, c, input_length, num_channels,
        input_start, kernel_size, dilation,
        max_val, max_idx, shared_data
    );
    
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
    bool return_indices
) {
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

    const dim3 blocks(
        (output_length + 31) / 32,
        (num_channels + 3) / 4,
        batch_size
    );
    const dim3 threads(32, 4);

    optimized_max_pool1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Optimized MaxPool1D forward (CUDA)");
}