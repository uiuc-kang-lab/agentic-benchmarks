#include <torch/extension.h>
#include <vector>

template <typename T>
__global__ void warp_aligned_argmax_kernel(
    const T* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) 
{
    extern __shared__ __align__(sizeof(float2)) unsigned char shmem[];
    float* sdata = reinterpret_cast<float*>(shmem);
    int* idata = reinterpret_cast<int*>(sdata + blockDim.x);

    const int tid = threadIdx.x;
    const int wid = tid >> 5;  // Warp ID
    const int lane = tid & 31; // Lane within warp
    const int k = blockIdx.x;
    
    if (k >= outerSize * innerSize) return;

    const int outer_idx = k / innerSize;
    const int inner_idx = k % innerSize;
    const int base_offset = (outer_idx * dimSize * innerSize) + inner_idx;

    // Initialize per-thread max values
    float max_val = -INFINITY;
    int max_idx = 0;

    // Each thread processes elements strided by warp size to ensure coalesced memory access
    #pragma unroll 4
    for (int d = lane; d < dimSize; d += 32) {
        float val = static_cast<float>(x[base_offset + d * innerSize]);
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }

    // Warp-level reduction using shuffle operations
    #pragma unroll 5
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        
        if (other_val > max_val || (other_val == max_val && other_idx < max_idx)) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        sdata[wid] = max_val;
        idata[wid] = max_idx;
    }
    __syncthreads();

    // Final reduction across warps (only first warp)
    if (wid == 0 && lane < (blockDim.x >> 5)) {
        max_val = sdata[lane];
        max_idx = idata[lane];
        
        #pragma unroll
        for (int offset = (blockDim.x >> 6); offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            
            if (other_val > max_val || (other_val == max_val && other_idx < max_idx)) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane == 0) {
            indices[k] = static_cast<int64_t>(max_idx);
        }
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim");

    int outerSize = 1, dimSize = sizes[dim], innerSize = 1;
    for (int d = 0; d < dim; ++d) outerSize *= sizes[d];
    for (int d = dim + 1; d < ndim; ++d) innerSize *= sizes[d];

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; ++d)
        if (d != dim) out_sizes.push_back(sizes[d]);
    
    auto indices = torch::empty(out_sizes, x.options().dtype(torch::kLong));
    
    const int threads = 128; // Multiple of warp size (32)
    const int blocks = outerSize * innerSize;
    const size_t smem = threads * sizeof(float) + threads * sizeof(int);
    
    warp_aligned_argmax_kernel<float><<<blocks, threads, smem>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );
    
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (warp aligned)");
}