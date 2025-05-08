#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assigns one block per pooling output and uses shared memory
// along with warp-level primitives to efficiently reduce the pooling window.
// Each block computes the maximum value (and index if required) for one output element.

__global__ void max_pool1d_shared_warp_kernel(
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

    // Each block handles one output pooling element
    int out_idx = blockIdx.x;
    int total_outputs = batch_size * num_channels * output_length;
    if (out_idx >= total_outputs) return;

    // Decode the flattened index into batch (b), channel (c) and output position (o)
    int o = out_idx % output_length;
    int tmp = out_idx / output_length;
    int c = tmp % num_channels;
    int b = tmp / num_channels;

    // Compute the starting position in the input for the pooling window
    int input_start = o * stride - padding;
    int base_idx = b * num_channels * input_length + c * input_length;

    // Each thread in the block computes a partial maximum over a subset of the pooling window
    float local_max = -INFINITY;
    int local_idx = -1;
    for (int j = threadIdx.x; j < kernel_size; j += blockDim.x) {
        int pos = input_start + j * dilation;
        if (pos >= 0 && pos < input_length) {
            float val = input[base_idx + pos];
            if (val > local_max) {
                local_max = val;
                local_idx = pos;
            }
        }
    }

    // Perform warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;  // threadIdx.x / 32
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(mask, local_max, offset);
        int other_idx = __shfl_down_sync(mask, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Each warp's lane 0 holds the partial result for that warp. Store these in shared memory.
    __shared__ float warp_max[32];  // Maximum number of warps per block assumed <= 32
    __shared__ int warp_idx[32];
    if (lane == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Let the first warp perform the final reduction over warp results
    int num_warps = (blockDim.x + 31) / 32;
    float final_max = -INFINITY;
    int final_idx = -1;
    if (threadIdx.x < num_warps) {
        final_max = warp_max[threadIdx.x];
        final_idx = warp_idx[threadIdx.x];
    }
    
    if (threadIdx.x < 32) {  // First warp does the final reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(mask, final_max, offset);
            int other_idx = __shfl_down_sync(mask, final_idx, offset);
            if (other_val > final_max) {
                final_max = other_val;
                final_idx = other_idx;
            }
        }
        if (threadIdx.x == 0) {
            output[out_idx] = final_max;
            if (return_indices) {
                indices[out_idx] = final_idx;
            }
        }
    }
}

// Host function that sets up and launches the CUDA kernel
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

    int total_outputs = batch_size * num_channels * output_length;
    // Launch one block per pooling output element
    int blockSize = 64; // 64 threads per block (multiple warps) for intra-block reduction
    int gridSize = total_outputs;

    max_pool1d_shared_warp_kernel<<<gridSize, blockSize>>>(
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
        return_indices);

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with shared memory and warp-level reductions (CUDA)");
}
