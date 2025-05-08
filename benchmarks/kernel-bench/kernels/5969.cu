#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// CUDA kernel for 1D Average Pooling using shared memory and warp-level reduction
// Each block computes one output element.
__global__ void shared_reduce_avg_pool1d_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    // Determine the output element index from grid dimensions
    int o = blockIdx.x;          // output spatial index
    int channel = blockIdx.y;      // channel index
    int batch = blockIdx.z;        // batch index

    // Compute output index in flattened tensor
    int out_idx = batch * in_channels * output_length + channel * output_length + o;

    // Compute the starting index in input for this pooling window
    int start = o * stride - padding;
    int base = batch * in_channels * input_length + channel * input_length;

    // Each thread accumulates a partial sum over a subset of the kernel window
    float sum = 0.0f;
    for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
        int pos = start + i;
        if (pos >= 0 && pos < input_length) {
            sum += input[base + pos];
        }
    }

    // Perform warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Shared memory for storing per-warp results
    extern __shared__ float shared_sum[];
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared_sum[warpId] = sum;
    }
    __syncthreads();

    // First warp reduces the partial sums from each warp
    int nWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (threadIdx.x < nWarps) {
        sum = shared_sum[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (threadIdx.x == 0) {
            // Write the result: average value (sum divided by kernel_size)
            output[out_idx] = sum / kernel_size;
        }
    }
}

// Host function
torch::Tensor shared_reduce_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor.");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Each block computes one output element; grid dims: (output_length, in_channels, batch_size)
    dim3 grid(output_length, in_channels, batch_size);
    // Choose a block size. 128 threads per block is typical.
    int blockSize = 128;
    
    // Calculate shared memory size: one float per warp
    int nWarps = (blockSize + WARP_SIZE - 1) / WARP_SIZE;
    size_t sharedMemSize = nWarps * sizeof(float);
    
    shared_reduce_avg_pool1d_kernel<<<grid, blockSize, sharedMemSize>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_reduce_avg_pool1d_forward, "1D Average Pooling forward (CUDA) with shared memory reduction");
}
