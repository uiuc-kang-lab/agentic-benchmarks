#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each block computes one output element (one pooling window) using intra-block reduction in shared memory.
// By assigning one output element to a block, we eliminate the need for global atomic operations. 
// Threads in the block collaboratively sum the pooling window values, then thread 0 writes the average result.

__global__ void avg_pool3d_forward_kernel_reduce(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Each block computes one output element
    int out_idx = blockIdx.x; // out_idx ranges over [0, total_output_elements)

    // Decode out_idx into (n, c, d_out, h_out, w_out)
    int w_out = out_idx % out_w;
    int tmp = out_idx / out_w;
    int h_out = tmp % out_h;
    tmp /= out_h;
    int d_out = tmp % out_d;
    tmp /= out_d;
    int c = tmp % channels;
    int n = tmp / channels;

    // Compute the starting indices of the pooling window in the input tensor
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // The pooling window always covers kernel_size^3 elements for count_include_pad=True
    int pool_volume = kernel_size * kernel_size * kernel_size;

    float sum = 0.0f;
    // Each thread in the block accumulates a partial sum over a portion of the pooling window.
    // The pooling window is linearized into [0, pool_volume), and threads stride over this range.
    for (int i = threadIdx.x; i < pool_volume; i += blockDim.x) {
        int di = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int hi = rem / kernel_size;
        int wi = rem % kernel_size;

        int input_d = d_start + di;
        int input_h = h_start + hi;
        int input_w = w_start + wi;

        // Only add valid indices from the input (else add 0)
        if (input_d >= 0 && input_d < in_d &&
            input_h >= 0 && input_h < in_h &&
            input_w >= 0 && input_w < in_w) {
            int input_index = (((n * channels + c) * in_d + input_d) * in_h + input_h) * in_w + input_w;
            sum += input[input_index];
        }
    }

    // Use shared memory to perform block-level reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes back the final result
    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0] / static_cast<float>(pool_volume);
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Total number of output elements
    int total_outputs = batch_size * channels * out_d * out_h * out_w;
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Launch one block per output element, with a fixed number of threads per block.
    int threads = 128; // Adjust based on pooling window size and hardware occupancy.
    dim3 grid(total_outputs);
    size_t shared_mem_size = threads * sizeof(float);

    avg_pool3d_forward_kernel_reduce<<<grid, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with block-level reduction");
}
