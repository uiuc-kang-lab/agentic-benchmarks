#include <torch/extension.h>
#include <vector>
#include <float.h>

// Constant memory for small dimension sizes
__constant__ int c_dimSize;

__global__ void hybrid_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int innerSize) {
    
    int slice = blockIdx.x;
    if (slice >= outerSize * innerSize) return;

    int outer_idx = slice / innerSize;
    int inner_idx = slice % innerSize;
    int base_offset = outer_idx * (c_dimSize * innerSize) + inner_idx;

    // Shared memory for reduction
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

    // Initial values
    float local_max = -FLT_MAX;
    int local_argmax = 0;

    // Coalesced memory access with unrolling for small dimensions
    if (c_dimSize <= 32) {
        #pragma unroll
        for (int d = threadIdx.x; d < c_dimSize; d += blockDim.x) {
            float val = x[base_offset + d * innerSize];
            if (val > local_max) {
                local_max = val;
                local_argmax = d;
            }
        }
    } else {
        // Strided access for larger dimensions
        for (int d = threadIdx.x; d < c_dimSize; d += blockDim.x) {
            float val = x[base_offset + d * innerSize];
            if (val > local_max) {
                local_max = val;
                local_argmax = d;
            }
        }
    }

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_argmax;
    __syncthreads();

    // Two-phase reduction: parallel reduction followed by warp-level reduction
    for (int offset = blockDim.x/2; offset >= 64; offset >>= 1) {
        if (threadIdx.x < offset) {
            if (s_max[threadIdx.x + offset] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + offset];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed)
    if (threadIdx.x < 32) {
        volatile float* vs_max = s_max;
        volatile int* vs_idx = s_idx;
        
        if (vs_max[threadIdx.x + 32] > vs_max[threadIdx.x]) {
            vs_max[threadIdx.x] = vs_max[threadIdx.x + 32];
            vs_idx[threadIdx.x] = vs_idx[threadIdx.x + 32];
        }
        if (vs_max[threadIdx.x + 16] > vs_max[threadIdx.x]) {
            vs_max[threadIdx.x] = vs_max[threadIdx.x + 16];
            vs_idx[threadIdx.x] = vs_idx[threadIdx.x + 16];
        }
        if (vs_max[threadIdx.x + 8] > vs_max[threadIdx.x]) {
            vs_max[threadIdx.x] = vs_max[threadIdx.x + 8];
            vs_idx[threadIdx.x] = vs_idx[threadIdx.x + 8];
        }
        if (vs_max[threadIdx.x + 4] > vs_max[threadIdx.x]) {
            vs_max[threadIdx.x] = vs_max[threadIdx.x + 4];
            vs_idx[threadIdx.x] = vs_idx[threadIdx.x + 4];
        }
        if (vs_max[threadIdx.x + 2] > vs_max[threadIdx.x]) {
            vs_max[threadIdx.x] = vs_max[threadIdx.x + 2];
            vs_idx[threadIdx.x] = vs_idx[threadIdx.x + 2];
        }
        if (vs_max[threadIdx.x + 1] > vs_max[threadIdx.x]) {
            vs_max[threadIdx.x] = vs_max[threadIdx.x + 1];
            vs_idx[threadIdx.x] = vs_idx[threadIdx.x + 1];
        }
    }

    if (threadIdx.x == 0) {
        indices[slice] = s_idx[0];
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Store dimension size in constant memory
    cudaMemcpyToSymbol(c_dimSize, &dimSize, sizeof(int));

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int slices = outerSize * innerSize;
    
    // Dynamic block size based on dimension size
    const int threads = (dimSize <= 512) ? 128 : 256;
    int blocks = slices;
    int sharedMemSize = threads * (sizeof(float) + sizeof(int));

    hybrid_argmax_kernel<<<blocks, threads, sharedMemSize>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "Hybrid ArgMax CUDA implementation");
}