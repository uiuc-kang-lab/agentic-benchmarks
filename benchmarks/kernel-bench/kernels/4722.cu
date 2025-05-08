#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Kernel 1: Each block computes a partial sum of squares and writes its result to a unique index in the output array.
__global__ void compute_norm_kernel_no_atomic(const float* input, float* partial_sums, int numel) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Stride loop over the input
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = sum;
    __syncthreads();

    // In-block reduction using shared memory
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's result to global memory (no atomics needed since each block writes to a unique index)
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Reduction kernel to reduce an array of floats to a smaller array, using a two-elements per thread approach
__global__ void reduce_partial_sums_kernel(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;

    if (idx < n) {
        sum = in[idx];
        if (idx + blockDim.x < n) {
            sum += in[idx + blockDim.x];
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    // In-block reduction
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// Kernel 3: Normalization kernel dividing each element by the Frobenius norm
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// Host function interfacing with PyTorch
torch::Tensor forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor
    auto output = torch::empty_like(input);
    int numel = input.numel();
    const int threads = 256;
    int blocks = std::min(65535, (numel + threads - 1) / threads);

    // Allocate tensor to hold partial sums from each block
    auto partial_sums = torch::empty({blocks}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* partial_ptr = partial_sums.data_ptr<float>();

    // Launch kernel to compute per-block sum of squares (no atomic operations used)
    compute_norm_kernel_no_atomic<<<blocks, threads>>>(input_ptr, partial_ptr, numel);

    // Iteratively reduce the partial sums to a single value
    int n = blocks;
    while (n > 1) {
        int threads_reduce = 256;
        int blocks_reduce = (n + threads_reduce * 2 - 1) / (threads_reduce * 2);

        auto temp = torch::empty({blocks_reduce}, input.options());
        float* temp_ptr = temp.data_ptr<float>();

        reduce_partial_sums_kernel<<<blocks_reduce, threads_reduce>>>(partial_ptr, temp_ptr, n);

        // Prepare for next iteration
        n = blocks_reduce;
        partial_sums = temp;
        partial_ptr = partial_sums.data_ptr<float>();
    }

    // Retrieve the final sum of squares and compute the Frobenius norm
    float sum_of_squares;
    cudaMemcpy(&sum_of_squares, partial_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    float norm = sqrt(sum_of_squares);

    // Launch normalization kernel to divide input elements by the computed norm
    int norm_blocks = std::min(65535, (numel + threads - 1) / threads);
    float* output_ptr = output.data_ptr<float>();
    normalize_kernel<<<norm_blocks, threads>>>(input_ptr, output_ptr, norm, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization using block reduction without global atomics");
}
