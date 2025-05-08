#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assigns one block per output element (for a given batch and channel).
// Each thread in the block computes a partial sum over a subset of the pooling window,
// then uses shared memory for intra-block reduction followed by warp-level final reduction using __shfl_down_sync().

__global__ void avg_pool1d_shared_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length) {

    // Grid mapping: 
    //   blockIdx.x -> output position (o)
    //   blockIdx.y -> channel index
    //   blockIdx.z -> batch index
    int o = blockIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    // Compute the starting index in the input for the pooling window
    int start = o * stride - padding;

    // Each thread computes a partial sum across a subset of the pooling window
    float partial_sum = 0.0f;
    for (int k = threadIdx.x; k < kernel_size; k += blockDim.x) {
        int pos = start + k;
        if (pos >= 0 && pos < input_length) {
            // Global index: batch * (in_channels * input_length) + channel * input_length + pos
            int input_index = batch * (gridDim.y * input_length) + channel * input_length + pos;
            partial_sum += input[input_index];
        }
    }

    // Allocate shared memory for intra-block reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Intra-block tree reduction using shared memory until 32 threads remain
    int tid = threadIdx.x;
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction using __shfl_down_sync
    if (tid < 32) {
        // Use volatile pointer for warp-synchronous access
        volatile float* vsdata = sdata;
        float sum = vsdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            sdata[0] = sum;
        }
    }
    __syncthreads();

    // Thread 0 writes the final averaged result to global memory
    if (threadIdx.x == 0) {
        // Global output index: batch * (in_channels * output_length) + channel * output_length + o
        int output_index = batch * (gridDim.y * gridDim.x) + channel * gridDim.x + o;
        output[output_index] = sdata[0] / kernel_size;
    }
}


// Forward function wrapper
torch::Tensor avg_pool1d_forward(
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

    // Set grid dimensions:
    //   grid.x = output_length, grid.y = in_channels, grid.z = batch_size
    dim3 grid(output_length, in_channels, batch_size);

    // Use a fixed number of threads per block for reduction; e.g., 128
    int threads_per_block = 128;
    dim3 block(threads_per_block);

    // Allocate shared memory: one float per thread
    int shared_mem_size = threads_per_block * sizeof(float);

    avg_pool1d_shared_kernel<<<grid, block, shared_mem_size>>>(
         x.data_ptr<float>(),
         output.data_ptr<float>(),
         kernel_size,
         stride,
         padding,
         input_length
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with shared memory and warp-level reduction (CUDA)");
}
