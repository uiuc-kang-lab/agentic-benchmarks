#include <torch/extension.h>
#include <vector>
#include <float.h>

// This kernel dynamically allocates threads per block based on the reduction dimension (dimSize).
// Each block is responsible for one slice (an outer and inner pair).
// Threads in the block cooperatively compute the maximum using a grid-stride loop over the reduction axis,
// and then perform a tree reduction in shared memory. When dimSize is small, fewer threads are launched to avoid
// underutilization; when it is large, up to 256 threads will share the work evenly.

__global__ void argmax_kernel_dynamic(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {
    // Each block processes one slice: an (outer, inner) pair
    int slice = blockIdx.x;
    if (slice >= outerSize * innerSize) return;

    int outer_idx = slice / innerSize;
    int inner_idx = slice % innerSize;
    int base_offset = outer_idx * (dimSize * innerSize) + inner_idx;

    // Each thread processes a subset of the reduction dimension via grid-stride loop
    float local_max = -FLT_MAX;
    int local_argmax = 0;
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float val = x[base_offset + d * innerSize];
        if (val > local_max) {
            local_max = val;
            local_argmax = d;
        }
    }

    // Allocate shared memory for reduction across threads in the block
    extern __shared__ char shm[];
    float* s_max = reinterpret_cast<float*>(shm);
    int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_argmax;
    __syncthreads();

    // Standard tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (s_max[threadIdx.x + stride] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // The first thread writes the result for this slice
    if (threadIdx.x == 0) {
        indices[slice] = s_idx[0];
    }
}

// Host function to launch the CUDA kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute outerSize (product of dimensions before 'dim'), dimSize, and innerSize (product of dimensions after 'dim')
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // The output tensor shape is the input shape with dimension 'dim' removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Each slice corresponds to one (outer, inner) pair
    int slices = outerSize * innerSize;

    // Choose the block size dynamically: use min(dimSize, 256) threads per block to match the reduction workload
    int block_size = (dimSize < 256 ? dimSize : 256);

    // Calculate optimal grid size based on device properties
    const int max_blocks_per_sm = 32;  // Typical maximum blocks per SM for compute capability 3.0+
    const int device_id = 0;  // Using first GPU
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    
    // Calculate target number of blocks based on device multiprocessors and occupancy
    int num_sms = props.multiProcessorCount;
    int target_blocks = num_sms * max_blocks_per_sm;
    
    // Ensure we have at least one block per slice, but don't exceed target blocks
    int blocks_per_slice = 1;
    int total_blocks = (slices + blocks_per_slice - 1) / blocks_per_slice;
    
    // Adjust total blocks to not exceed target while maintaining coverage
    if (total_blocks > target_blocks) {
        blocks_per_slice = (slices + target_blocks - 1) / target_blocks;
        total_blocks = (slices + blocks_per_slice - 1) / blocks_per_slice;
    }
    
    dim3 grid(total_blocks);
    dim3 block(block_size);
    size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));

    argmax_kernel_dynamic<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward with dynamic thread allocation");
}
