#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory and warp-level primitives for reduction
__global__ void shfl_shared_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    // Each block processes one output element
    const int total_outputs = batch_size * in_channels * output_length;
    const int out_index = blockIdx.x;
    if (out_index >= total_outputs) return;

    // Decode output element indices
    const int o = out_index % output_length;
    const int channel = (out_index / output_length) % in_channels;
    const int batch = out_index / (output_length * in_channels);

    const int input_base = batch * in_channels * input_length + channel * input_length;
    const int start_idx = o * stride - padding;

    float sum = 0.0f;
    // Each thread computes a partial sum over the kernel window
    for (int k = threadIdx.x; k < kernel_size; k += blockDim.x) {
        int pos_input = start_idx + k;
        float val = 0.0f;
        if (pos_input >= 0 && pos_input < input_length) {
            val = input[input_base + pos_input];
        }
        sum += val;
    }

    // Intra-warp reduction using warp shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write each warp's result to shared memory
    __shared__ float shared[32];  // limit: one value per warp
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // First warp loads the partial sums from shared memory and reduces
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    sum = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (threadIdx.x < warpSize) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
    }

    // Thread 0 writes the final average for this output element
    if (threadIdx.x == 0) {
        output[out_index] = sum / kernel_size;
    }
}

// Host function to launch the kernel
torch::Tensor shfl_shared_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());
    
    const int total_outputs = batch_size * in_channels * output_length;
    // Launch one block per output element; use 256 threads per block
    const int threads = 256;
    dim3 blocks(total_outputs);
    dim3 threadBlock(threads);

    shfl_shared_avg_pool1d_kernel<<<blocks, threadBlock>>>(
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
    m.def("forward", &shfl_shared_avg_pool1d_forward, "Shuffle and Shared Memory 1D Average Pooling forward (CUDA)");
}
