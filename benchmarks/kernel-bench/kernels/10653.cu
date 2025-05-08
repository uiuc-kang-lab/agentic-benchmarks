#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel computes the cumulative product along a given dimension using a parallel scan (prefix product) algorithm.
// Each independent cumulative product sequence is processed by one block. The block's shared memory holds the elements
// (padded to the next power of two) which are then processed using the Blelloch scan algorithm to compute an exclusive scan.
// Finally, the inclusive cumulative product is obtained by multiplying the exclusive scan result by the original element.
// This approach distributes the workload evenly among the threads in each block, helping to reduce runtime.

template <typename scalar_t>
__global__ void cumprod_kernel_parallel_scan(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_sequences) {

    // Each block handles one independent cumulative product sequence.
    int seq_idx = blockIdx.x;
    if (seq_idx >= total_sequences) return;

    // Decode sequence index into batch index and inner index based on the original mapping:
    // The number of sequences is numel / dim_size, where each sequence corresponds to a unique (batch_idx, in_idx).
    int batch_idx = seq_idx / stride;
    int in_idx = seq_idx % stride;
    int64_t base_offset = batch_idx * (stride * dim_size) + in_idx;

    // 'n' is the next power-of-two greater than or equal to dim_size (blockDim.x is set to n).
    int n = blockDim.x;
    int tid = threadIdx.x;

    // Shared memory holds the elements for the scan.
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Load the input elements for this sequence into shared memory. For threads beyond dim_size, use the identity (1).
    if (tid < dim_size) {
        sdata[tid] = input[base_offset + tid * stride];
    } else {
        sdata[tid] = (scalar_t)1;
    }
    __syncthreads();

    // Up-sweep (reduce) phase to build the reduction tree.
    // At each step, pairs of elements are multiplied.
    for (int offset = 1; offset < n; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < n) {
            sdata[index] = sdata[index] * sdata[index - offset];
        }
        __syncthreads();
    }

    // Down-sweep phase to convert the reduction tree into an exclusive scan.
    if (tid == 0) {
        sdata[n - 1] = (scalar_t)1; // Set the last element to the identity.
    }
    __syncthreads();

    for (int offset = n / 2; offset >= 1; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < n) {
            scalar_t temp = sdata[index - offset];
            sdata[index - offset] = sdata[index];
            sdata[index] = temp * sdata[index];
        }
        __syncthreads();
    }

    // Convert the exclusive scan to an inclusive scan by multiplying with the original input element.
    if (tid < dim_size) {
        scalar_t orig = input[base_offset + tid * stride];
        output[base_offset + tid * stride] = sdata[tid] * orig;
    }
}


// Host function to launch the parallel scan kernel
torch::Tensor cumprod_cuda_forward_parallel_scan(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    // Retrieve tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    // Total number of independent sequences (each to be scanned) is numel / dim_size.
    int64_t total_sequences = numel / dim_size;

    // Compute the next power-of-two for dim_size.
    int n = 1;
    while (n < dim_size) n *= 2;

    // Launch one block per sequence, with n threads per block.
    const int threads = n;
    const int blocks = total_sequences;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_parallel_scan", ([&] {
        int shared_mem_size = threads * sizeof(scalar_t);
        cumprod_kernel_parallel_scan<scalar_t><<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride,
            total_sequences
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_parallel_scan, "Cumulative product forward parallel scan (CUDA)");
}
