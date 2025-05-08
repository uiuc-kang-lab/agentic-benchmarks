#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the reverse cumulative sum along the given dimension.
// It assumes that the cumsum dimension (n_cols) is not larger than the next power-of-two thread block size (max 1024).
// Each block processes one row of the input tensor. The elements along the row are first loaded in reverse order into shared memory,
// then an in-block parallel exclusive scan (Blelloch scan) is performed. Finally, the inclusive result is obtained by adding back the original
// reversed elements, and written back in reverse to produce the correct reverse cumulative sum.
// Note: No atomic operations are used because each row is handled by a single thread block, eliminating global race conditions.


// Kernel definition
__global__ void reverse_cumsum_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int n_cols) {
    // Each block processes one row
    int row = blockIdx.x;
    int row_start = row * n_cols;

    // Use dynamic shared memory allocated as two contiguous segments:
    // first for the working array for scan and second for storing the original reversed values
    extern __shared__ float shared_mem[];
    float* sdata = shared_mem;             // working array for scan
    float* orig  = shared_mem + blockDim.x; // to store original values

    int tid = threadIdx.x;

    // Load the row in reverse order into shared memory.
    // If n_cols is less than blockDim.x (the padded size), pad with 0.
    float val = (tid < n_cols) ? input[row_start + (n_cols - 1 - tid)] : 0.0f;
    sdata[tid] = val;
    orig[tid] = val;
    __syncthreads();

    // Perform upsweep phase of Blelloch scan (exclusive scan)
    // Assumption: blockDim.x is a power of two
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            sdata[index] += sdata[index - offset];
        }
        __syncthreads();
    }

    // Set the last element to 0 for downsweep
    if (tid == 0) {
        sdata[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Perform downsweep phase
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            float temp = sdata[index - offset];
            sdata[index - offset] = sdata[index];
            sdata[index] += temp;
        }
        __syncthreads();
    }

    // Compute the inclusive scan result by adding the original values.
    // Write the result back in reverse order to recover the desired output order:
    // out[i] = sum(input[i] ... input[n_cols-1]).
    if (tid < n_cols) {
        float inclusive = sdata[tid] + orig[tid];
        output[row_start + (n_cols - 1 - tid)] = inclusive;
    }
}


// Host function: reverse cumulative sum along the specified dimension.
// For simplicity, we assume that the operation is applied along a contiguous dimension (e.g. the last dimension).

autotensor::Tensor reverse_cumsum(at::Tensor input, int64_t dim) {
    // Ensure tensor is contiguous and on CUDA
    input = input.contiguous();
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    // For simplicity, assume that 'dim' is the cumulative dimension and the tensor is 2D or can be viewed as [num_rows, n_cols].
    int64_t n_cols = input.size(dim);
    int64_t numel = input.numel();
    int64_t num_rows = numel / n_cols;

    auto output = torch::empty_like(input);

    // Compute next power of two for thread block size
    auto nextPow2 = [](int n) {
        int p = 1;
        while (p < n) p *= 2;
        return p;
    };
    int threads = nextPow2(n_cols);
    TORCH_CHECK(threads <= 1024, "n_cols is too large to be processed by a single block");

    dim3 grid(num_rows);
    dim3 block(threads);
    size_t shared_mem_size = threads * 2 * sizeof(float); // shared memory for sdata and orig

    // Launch the kernel
    reverse_cumsum_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n_cols);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in reverse_cumsum_kernel: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Optimized reverse cumulative sum along a specified dimension (CUDA)");
}
