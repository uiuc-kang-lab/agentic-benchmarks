#include <torch/extension.h>
#include <vector>
#include <cfloat>
#include <cuda_runtime.h>

// Kernel: Each block computes an output element (one (outer, inner) pair) by reducing over the 'dim' dimension using a tunable block size.
__global__ void exp_blocksize_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Determine the (outer, inner) index from blockIdx.x
    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Allocate shared memory for reduction: first blockDim.x floats, then blockDim.x ints
    extern __shared__ float shared[];  // shared memory: first half for values, second half for indices
    int* s_idx = (int*)&shared[blockDim.x];

    // Each thread computes a local maximum over portions of the dim dimension
    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = __ldg(&x[base_offset + i * innerSize]);
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }

    // Store the per-thread results into shared memory
    shared[threadIdx.x] = thread_max;
    s_idx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                shared[threadIdx.x] = shared[threadIdx.x + s];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction without __syncthreads using volatile pointers
    if (threadIdx.x < 32) {
        volatile float* vsdata = shared;
        volatile int* vsidx = s_idx;
        for (int offset = 32; offset > 0; offset >>= 1) {
            if (threadIdx.x + offset < blockDim.x && vsdata[threadIdx.x + offset] > vsdata[threadIdx.x]) {
                vsdata[threadIdx.x] = vsdata[threadIdx.x + offset];
                vsidx[threadIdx.x] = vsidx[threadIdx.x + offset];
            }
        }
    }

    // Thread 0 writes the final argmax index for this output element
    if (threadIdx.x == 0) {
        indices[global_idx] = s_idx[0];
    }
}

// Host function: Automatically selects an optimal block size from candidate sizes {32, 64, 128, 256, 512}
// using cudaOccupancyMaxPotentialBlockSize and launches the kernel accordingly.

torch::Tensor exp_blocksize_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute sizes for the outer, reduced, and inner dimensions
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Prepare the output shape: the same as input with the 'dim' dimension removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Determine the total number of output elements
    int total_outputs = outerSize * innerSize;

    // Use cudaOccupancyMaxPotentialBlockSize to determine an optimal block size
    int minGridSize, optimalBlockSize;
    cudaError_t occError = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &optimalBlockSize, exp_blocksize_argmax_kernel, 0, 0);
    if (occError != cudaSuccess) {
        optimalBlockSize = 128;  // fallback value
    }

    // Candidate block sizes to experiment with
    int candidates[5] = {32, 64, 128, 256, 512};
    int chosenBlockSize = candidates[4];  // default to the largest candidate
    for (int i = 0; i < 5; i++) {
        if (optimalBlockSize <= candidates[i]) {
            chosenBlockSize = candidates[i];
            break;
        }
    }

    dim3 grid(total_outputs);
    dim3 block(chosenBlockSize);

    // Allocate shared memory: blockSize floats for values + blockSize ints for indices
    size_t shared_mem_size = chosenBlockSize * (sizeof(float) + sizeof(int));

    exp_blocksize_argmax_kernel<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &exp_blocksize_argmax_forward_cuda, "Experimental Block Size ArgMax CUDA forward");
}
