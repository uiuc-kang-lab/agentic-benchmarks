#include <torch/extension.h>
#include <vector>
#include <cfloat>

// This kernel ensures memory coalescing for global loads when possible.
// For the case innerSize == 1, the reduction elements are contiguous in memory.
// For innerSize > 1, we use __ldg to read the strided elements, but the fallback is still correct.

__global__ void coalesced_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Each block computes one output element corresponding to one (outer, inner) pair.
    int global_idx = blockIdx.x;
    int outer_idx, inner_idx;
    if(innerSize == 1) {
        // When innerSize is 1, the output grid size equals outerSize and data along dim are contiguous.
        outer_idx = global_idx;
        inner_idx = 0;
    } else {
        outer_idx = global_idx / innerSize;
        inner_idx = global_idx % innerSize;
    }

    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Allocate shared memory: first blockDim.x floats for the max values, then blockDim.x ints for the argmax indices
    extern __shared__ float sdata[];
    int* sidx = (int*)(&sdata[blockDim.x]);

    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    if(innerSize == 1) {
        // Coalesced read: elements are contiguous when innerSize==1
        for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
            float val = x[base_offset + i];
            if(val > thread_max) {
                thread_max = val;
                thread_max_idx = i;
            }
        }
    } else {
        // Non-coalesced read: use __ldg for cached load
        for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
            float val = __ldg(&x[base_offset + i * innerSize]);
            if(val > thread_max) {
                thread_max = val;
                thread_max_idx = i;
            }
        }
    }

    sdata[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        indices[global_idx] = sidx[0];
    }
}

// Host function to launch the kernel

torch::Tensor coalesced_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim.");

    // Compute outerSize, dimSize and innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Total number of outputs corresponds to the product of outerSize and innerSize
    int total_outputs = outerSize * innerSize;
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if(d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int blockSize = 128;  // Block size tuned for performance
    dim3 grid(total_outputs);
    dim3 block(blockSize);
    size_t shared_mem = blockSize * (sizeof(float) + sizeof(int));

    coalesced_argmax_kernel<<<grid, block, shared_mem>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_argmax_forward_cuda, "Coalesced ArgMax CUDA forward");
}
