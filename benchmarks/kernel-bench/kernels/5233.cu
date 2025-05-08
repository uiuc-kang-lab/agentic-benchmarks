#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

// Optimized kernel using block-level parallel reduction with shared memory
__global__ void max_pool1d_kernel_optimized(
    const float* input,
    float* output,
    int64_t* indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    const bool return_indices) {

    // Each block processes multiple output elements
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int channel = blockIdx.y;            // channel index
    int batch = blockIdx.z;              // batch index
    
    // Calculate number of output elements per block
    const int elements_per_block = 4;  // Process 4 elements per block for better efficiency
    int block_start = blockIdx.x * elements_per_block;
    
    // Process multiple output elements per block
    for (int element_offset = 0; element_offset < elements_per_block; element_offset++) {
        int out_idx = block_start + element_offset;
        if (out_idx >= output_length) break;  // Guard against out-of-bounds
        
        // Compute the starting index of the pooling window in the input
        int input_start = out_idx * stride - padding;

    // Each thread computes a partial maximum over a subset of the pooling window
    float local_max = -INFINITY;
    int local_argmax = -1;
    for (int k = tid; k < kernel_size; k += nthreads) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            float val = input[batch * num_channels * input_length + channel * input_length + pos];
            if (val > local_max) {
                local_max = val;
                local_argmax = pos;
            }
        }
    }

    // Allocate dynamic shared memory for reduction
    extern __shared__ char shared_memory[];
    float* sdata = reinterpret_cast<float*>(shared_memory);
    int* sindex = reinterpret_cast<int*>(shared_memory + nthreads * sizeof(float));

    sdata[tid] = local_max;
    sindex[tid] = local_argmax;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int s = nthreads / 2; s > 0; s /= 2) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }

    // The first thread writes the result to global memory
    if (tid == 0) {
        int out_index = batch * num_channels * output_length + channel * output_length + out_idx;
        output[out_index] = sdata[0];
        if (return_indices) {
            indices[out_index] = sindex[0];
        }
    }
}

// Forward function called from PyTorch
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

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
                                 options.dtype(torch::kInt64));
    }

    // Each block processes one pooling window (one output element) for a given channel and batch
    const dim3 blocks(output_length, num_channels, batch_size);
    const dim3 threads(BLOCK_SIZE);
    size_t shared_mem_size = BLOCK_SIZE * (sizeof(float) + sizeof(int));

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

    // Return concatenated output and indices if indices are requested
    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MaxPool1D forward (CUDA)");
}
