#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Kernel: Each block handles one output element (i.e. one (outer, inner) pair) and reduces over the 'dim' dimension
// using shared memory. Atomic operations in global memory are not used because each block independently computes
// its own output, thereby eliminating race conditions and minimizing global contention.

// The kernel assumes that each block is launched with a fixed number of threads (blockDim.x).
// Shared memory is allocated to hold both partial max values and their corresponding indices.

__global__ void per_output_block_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Each block is responsible for one output element.
    // Global index corresponds to the (outer, inner) pair. 
    int global_idx = blockIdx.x;

    // Decode outer and inner indices from global_idx given innerSize
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;

    // Compute the base offset for this slice in the input
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Allocate shared memory for reduction.
    // First part stores partial max values (floats), second part stores corresponding indices (ints).
    extern __shared__ float shared[]; // Total shared memory = blockDim.x * (sizeof(float)+sizeof(int)) bytes
    int* sidx = (int*)&shared[blockDim.x];

    // Each thread computes a partial maximum by processing a chunk of the dimension
    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    // Loop over the reduction dimension with stride equal to blockDim.x
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = x[base_offset + i * innerSize];
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }

    // Store each thread's partial results into shared memory
    shared[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Perform reduction in shared memory to find the maximum value and its index
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                shared[threadIdx.x] = shared[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // The first thread writes the final result for this output element
    if (threadIdx.x == 0) {
        indices[global_idx] = sidx[0];
    }
}

// Host function to launch the per-output-block argmax kernel

torch::Tensor per_output_block_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    // Ensure input tensor is contiguous
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute outerSize, dimSize, and innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Total number of output elements is outerSize * innerSize
    int total_outputs = outerSize * innerSize;

    // Prepare output shape: input shape with the reduced dimension removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if(d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Choose the number of threads per block (e.g., 128) for the reduction along the dim
    int blockSize = 128;
    // Launch one block per output element
    dim3 grid(total_outputs);
    dim3 block(blockSize);

    // Compute shared memory size: blockSize floats + blockSize ints
    size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));

    per_output_block_argmax_kernel<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &per_output_block_argmax_forward_cuda, "Per-output-block ArgMax CUDA forward");
}
