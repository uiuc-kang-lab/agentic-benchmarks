#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Device function to compute local maximum for a thread
__device__ __forceinline__ void compute_local_max(
    const float* __restrict__ x,
    const int base_offset,
    const int dimSize,
    const int innerSize,
    float& thread_max,
    int& thread_max_idx) {
    
    thread_max = -FLT_MAX;
    thread_max_idx = 0;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = __ldg(&x[base_offset + i * innerSize]);
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }
}

// Device function to perform shared memory reduction
__device__ __forceinline__ void reduce_shared_memory(
    float* shared,
    int* sidx,
    const int tid) {
    
    __syncthreads();
    
    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
}

// Device function to compute output indices
__device__ __forceinline__ void compute_output_indices(
    const int global_idx,
    const int innerSize,
    int& outer_idx,
    int& inner_idx,
    int& base_offset) {
    
    outer_idx = global_idx / innerSize;
    inner_idx = global_idx % innerSize;
    base_offset = outer_idx * innerSize;
}

__global__ void modular_device_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    extern __shared__ float shared[];
    int* sidx = (int*)&shared[blockDim.x];
    
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x;
    
    int outer_idx, inner_idx, base_offset;
    compute_output_indices(global_idx, innerSize, outer_idx, inner_idx, base_offset);
    
    base_offset = outer_idx * dimSize * innerSize + inner_idx;
    
    float thread_max;
    int thread_max_idx;
    compute_local_max(x, base_offset, dimSize, innerSize, thread_max, thread_max_idx);
    
    shared[tid] = thread_max;
    sidx[tid] = thread_max_idx;
    
    reduce_shared_memory(shared, sidx, tid);
    
    if (tid == 0) {
        indices[global_idx] = sidx[0];
    }
}

torch::Tensor modular_device_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) if (d != dim) out_sizes.push_back(sizes[d]);
    auto indices = torch::empty(out_sizes, torch::TensorOptions().device(x.device()).dtype(torch::kLong));

    int blockSize = 128;
    dim3 grid(outerSize * innerSize);
    size_t shared_mem = blockSize * (sizeof(float) + sizeof(int));

    modular_device_argmax_kernel<<<grid, blockSize, shared_mem>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_device_argmax_forward_cuda, "Modular Device ArgMax CUDA forward");
}