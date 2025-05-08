#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assigns one output element per block. Partial sums along the reduction dimension are computed by each thread and stored in shared memory.
// The reduction in shared memory uses __syncthreads() only where necessary to ensure consistency, and then the final warp-level reduction is performed without additional synchronizations.

template <typename scalar_t>
__global__ void block_shared_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block computes one output element
    int out_idx = blockIdx.x;
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread accumulates a partial sum over the reduction dimension
    scalar_t sum = 0;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base + i * inner_size];
    }

    // Allocate shared memory (dynamically sized) as a raw byte array and cast to scalar_t*
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    sdata[threadIdx.x] = sum;
    __syncthreads();  // Ensure all partial sums are written before reduction

    // Perform tree-based reduction in shared memory.
    // __syncthreads() is used only when required for shared memory consistency.
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction; no __syncthreads() needed since threads in a warp are implicitly synchronized
    if (threadIdx.x < 32) {
        volatile scalar_t* vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
        if (threadIdx.x == 0) {
            output[out_idx] = vsmem[0];
        }
    }
}

// CUDA wrapper function
// For each output element (of shape outer_size x inner_size), one block is launched to perform the reduction.
// This kernel minimizes __syncthreads() usage by limiting it to shared memory consistency only.

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Normalize dimension
    if (dim < 0) dim += input.dim();

    // Calculate sizes
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // The output tensor has the reduced dimension set to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t total_outputs = outer_size * inner_size;

    // Launch configuration: one block per output element
    int threads = 256;   // Block size
    int blocks = total_outputs;  

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        size_t shared_mem_bytes = threads * sizeof(scalar_t);
        block_shared_sum_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) with minimal __syncthreads() usage");
}
