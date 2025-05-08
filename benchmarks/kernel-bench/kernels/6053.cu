#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel inverts the pooling computation: each thread processes a subset of input elements
// and "scatters" its contribution (input[j] / kernel_size) to all output elements whose pooling window
// includes that input. The accumulation is done in shared memory using atomicAdd, which is fast
// compared to global atomics. Each block handles one (batch, channel) pair, so there is no inter-block
// race. Finally, one thread writes the shared memory result to global output.

__global__ void avg_pool1d_scatter_kernel(
    const float *__restrict__ input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    // Each block processes one (batch, channel) pair.
    int channel = blockIdx.x;
    int batch = blockIdx.y;

    // Pointers to the specific input and output for this (batch, channel).
    const float *in_ptr = input + batch * in_channels * input_length + channel * input_length;
    float *out_ptr = output + batch * in_channels * output_length + channel * output_length;

    // Allocate shared memory buffer for accumulating output sums.
    extern __shared__ float s_out[];

    // Initialize shared memory to 0 in parallel.
    for (int i = threadIdx.x; i < output_length; i += blockDim.x) {
        s_out[i] = 0.0f;
    }
    __syncthreads();

    // Each thread processes a subset of input indices with a strided loop.
    for (int j = threadIdx.x; j < input_length; j += blockDim.x) {
        float val = in_ptr[j];
        
        // Determine the range of output indices whose pooling window includes input index j.
        // The condition for output index i is: i*stride - padding <= j <= i*stride - padding + kernel_size - 1
        // Solving for i gives:
        //   i >= ceil((j + padding - kernel_size + 1) / stride)
        //   i <= floor((j + padding) / stride)
        int numerator = j + padding - kernel_size + 1;
        int i_min = (numerator > 0) ? ((numerator + stride - 1) / stride) : 0;
        int i_max = (j + padding) / stride;  // floor division
        if (i_max >= output_length) {
            i_max = output_length - 1;
        }
        
        // Scatter the normalized contribution to all valid output indices.
        if (i_min <= i_max) {
            float contribution = val / kernel_size;
            for (int i = i_min; i <= i_max; ++i) {
                atomicAdd(&s_out[i], contribution);
            }
        }
    }
    __syncthreads();

    // Write the accumulated shared memory output to global memory.
    for (int i = threadIdx.x; i < output_length; i += blockDim.x) {
        out_ptr[i] = s_out[i];
    }
}


torch::Tensor avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Launch one block per (channel, batch) pair.
    dim3 grid(in_channels, batch_size);
    int blockSize = 128;  // Use 256 threads per block
    int shared_mem_size = output_length * sizeof(float);

    avg_pool1d_scatter_kernel<<<grid, blockSize, shared_mem_size>>>(
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
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with scatter and shared memory atomics (CUDA)");
}
