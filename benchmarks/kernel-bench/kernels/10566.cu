#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel assumes that the cumulative product is performed along the last (contiguous) dimension.
// It loads each row (i.e. each independent cumulative-product sequence) into shared memory using coalesced global memory accesses,
// performs the cumulative product sequentially in shared memory (which is very fast),
// and then writes the result back to global memory in a coalesced manner.

template <typename scalar_t>
__global__ void cumprod_coalesced_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t row_length,
    const int64_t total_batches) {

    // Each block processes one row (one cumulative product sequence).
    int batch = blockIdx.x;
    if (batch >= total_batches) return;
    
    // Calculate the base index for this row in the flattened array
    int64_t base = batch * row_length;

    // Allocate shared memory for one row. Use extern char[] and reinterpret cast to allow dynamic shared memory allocation.
    extern __shared__ char shared_mem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

    int tid = threadIdx.x;
    
    // Load the row from global memory into shared memory in a coalesced manner
    for (int i = tid; i < row_length; i += blockDim.x) {
        sdata[i] = input[base + i];
    }
    __syncthreads();

    // Perform the cumulative product sequentially in shared memory.
    // Only one thread (e.g. thread 0) does this, as the operation is inherently sequential,
    // but shared memory access is very fast.
    if (tid == 0) {
        for (int i = 1; i < row_length; i++) {
            sdata[i] = sdata[i - 1] * sdata[i];
        }
    }
    __syncthreads();

    // Write the results back to global memory in a coalesced manner
    for (int i = tid; i < row_length; i += blockDim.x) {
        output[base + i] = sdata[i];
    }
}

// Forward function
// This function assumes that the cumulative product dimension is contiguous. If not, it makes the tensor contiguous.
// For best performance, it is recommended to perform the cumulative product along the last dimension.

torch::Tensor cumprod_coalesced_forward(torch::Tensor input, int64_t dim) {
    // Ensure the tensor is contiguous. For full generality, one could transpose the tensor so that the dimension 'dim'
    // becomes the last dimension, but here we assume the user provides a tensor with the cumulative dimension contiguous.
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto sizes = input.sizes();
    // For coalescing, we assume the cumulative product is along the last dimension.
    int64_t row_length = sizes[dim];
    int64_t total_batches = input.numel() / row_length;

    auto output = torch::empty_like(input);

    // Choose block size: use up to 256 threads per block (you can adjust this based on row_length)
    int block_size = (row_length < 256) ? row_length : 256;
    int grid_size = total_batches;

    // Launch the kernel with dynamic shared memory size equal to row_length * sizeof(scalar_t)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_coalesced", ([&] {
        cumprod_coalesced_kernel<scalar_t><<<grid_size, block_size, row_length * sizeof(scalar_t)>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            row_length,
            total_batches
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_coalesced_forward, "Cumulative product forward (CUDA, coalesced)");
}
