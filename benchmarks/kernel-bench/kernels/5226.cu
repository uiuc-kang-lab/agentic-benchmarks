#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// This kernel processes a contiguous segment of the output row for each (batch, channel) pair
// ensuring that global memory writes are coalesced. It loads the required segment of the input into
// shared memory in a coalesced manner, so that the pooling window accesses read from shared memory,
// improving bandwidth utilization.

__global__ void coalesced_max_pool1d_kernel(
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
    const bool return_indices) {

    // Each block handles one (batch, channel) row segment
    int bc = blockIdx.x;  // bc ranges over batch_size * num_channels
    int b = bc / num_channels;
    int c = bc % num_channels;

    // Determine the segment of output indices processed by this block
    int out_offset = blockIdx.y * blockDim.x;  // starting output index for this block
    int tid = threadIdx.x;
    int global_out_idx = out_offset + tid;  // overall output index in the row

    // Base pointers for the (b, c) row
    const float* input_row = input + (b * num_channels + c) * input_length;
    float* output_row = output + (b * num_channels + c) * output_length;
    int64_t* indices_row = return_indices ? (indices + (b * num_channels + c) * output_length) : nullptr;

    // Compute the required size for shared memory for this block
    // For a full block (threads), the shared region covers the range of input needed
    // for output indices [out_offset, out_offset + blockDim.x - 1].
    // Starting global input index for the block:
    int global_input_start = out_offset * stride - padding;
    // The shared memory size is computed as the number of input elements needed:
    // ((blockDim.x - 1) * stride + (kernel_size - 1) * dilation + 1).
    int smem_size = (blockDim.x - 1) * stride + (kernel_size - 1) * dilation + 1;

    extern __shared__ float sdata[];

    // Load required input region into shared memory in a coalesced manner
    // Each thread loads multiple elements if necessary
    for (int i = tid; i < smem_size; i += blockDim.x) {
        int global_input_idx = global_input_start + i;
        if (global_input_idx < 0 || global_input_idx >= input_length) {
            sdata[i] = -INFINITY;
        } else {
            sdata[i] = input_row[global_input_idx];
        }
    }
    __syncthreads();

    // Only process if the global output index is within bounds
    if (global_out_idx < output_length) {
        // For the output element, compute the global start of its pooling window
        int pool_global_start = global_out_idx * stride - padding;
        // Compute the offset into shared memory corresponding to pool_global_start
        int base = pool_global_start - global_input_start;
        float max_val = -INFINITY;
        int max_index = -1;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            int offset = base + k * dilation;
            float val = sdata[offset];
            if (val > max_val) {
                max_val = val;
                max_index = pool_global_start + k * dilation; // global index in input
            }
        }
        // Write output with coalesced global memory access
        output_row[global_out_idx] = max_val;
        if (return_indices) {
            indices_row[global_out_idx] = max_index;
        }
    }
}


// Host function that sets up grid/block dimensions and launches the kernel
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

    int batch_size = x.size(0);
    int num_channels = x.size(1);
    int input_length = x.size(2);

    int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Set the number of threads per block for processing the output row
    int threads = 256;
    // grid.x covers all (batch, channel) pairs
    // grid.y covers segments of the output length per row
    dim3 blocks(batch_size * num_channels, (output_length + threads - 1) / threads);
    dim3 block(threads);
    
    // Compute shared memory size per block
    int shared_mem_size = ((threads - 1) * stride + (kernel_size - 1) * dilation + 1) * sizeof(float);

    coalesced_max_pool1d_kernel<<<blocks, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Coalesced MaxPool1D forward (CUDA)");
}
