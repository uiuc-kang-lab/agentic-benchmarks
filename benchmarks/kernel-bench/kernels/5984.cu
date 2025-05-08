#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory to load a contiguous segment of the input for a contiguous block of output elements.
// This ensures that global memory accesses are coalesced when loading input and writing output.

__global__ void coalesced_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int in_channels) {

    // Each block processes a contiguous segment of output elements for one (batch, channel) pair.
    const int block_output_start = blockIdx.x * blockDim.x; // first output index for this block
    const int tid = threadIdx.x;
    const int batch = blockIdx.z;
    const int channel = blockIdx.y;

    // Compute base pointers for the current (batch, channel) pair
    const int input_base = batch * in_channels * input_length + channel * input_length;
    const int output_base = batch * in_channels * output_length + channel * output_length;

    // Determine the starting input index required by this block
    int in_start = block_output_start * stride - padding;
    // The shared memory region must cover indices for all outputs in this block.
    // For outputs [block_output_start, block_output_start + blockDim.x - 1], the maximum required index is:
    // ((block_output_start + blockDim.x - 1) * stride - padding) + (kernel_size - 1).
    // Thus, shared memory size is:
    int shared_len = (blockDim.x - 1) * stride + kernel_size;

    extern __shared__ float sh_data[];

    // Load the necessary input segment into shared memory cooperatively
    for (int i = tid; i < shared_len; i += blockDim.x) {
        int in_index = in_start + i;
        float val = 0.0f;
        if (in_index >= 0 && in_index < input_length) {
            val = input[input_base + in_index];
        }
        sh_data[i] = val;
    }
    __syncthreads();

    // Each thread computes one output element if within bounds
    int out_index = block_output_start + tid;
    if (out_index < output_length) {
        // Calculate the offset within the shared memory window for this output element
        int sh_offset = (out_index * stride - padding) - in_start;
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            sum += sh_data[sh_offset + k];
        }
        output[output_base + out_index] = sum / kernel_size;
    }
}

// Host function to launch the kernel
torch::Tensor coalesced_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Configure grid: each block processes a contiguous segment for one channel in one batch
    int threads = 256;
    int blocks_x = (output_length + threads - 1) / threads;
    dim3 blocks(blocks_x, in_channels, batch_size);
    dim3 threads_per_block(threads);

    // Shared memory size (in bytes): (blockDim.x - 1) * stride + kernel_size elements
    int shared_memory_size = ((threads - 1) * stride + kernel_size) * sizeof(float);

    coalesced_avg_pool1d_kernel<<<blocks, threads_per_block, shared_memory_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_avg_pool1d_forward, "Coalesced 1D Average Pooling forward (CUDA)");
}
