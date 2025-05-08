#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

template<typename T>
__global__ void argmax_aligned_memory_kernel(
    const T* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize)
{
    extern __shared__ __align__(sizeof(float2)) unsigned char shmem[];
    float2* sdata = reinterpret_cast<float2*>(shmem);
    
    const int k = blockIdx.x;
    if (k >= outerSize * innerSize) return;
    
    const int outer_idx = k / innerSize;
    const int inner_idx = k % innerSize;
    const int start_offset = (outer_idx * dimSize * innerSize) + inner_idx;
    
    float thread_max = -INFINITY;
    int thread_idx = -1;

    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float val = __ldg(&x[start_offset + d * innerSize]);
        if (val > thread_max || (val == thread_max && d < thread_idx)) {
            thread_max = val;
            thread_idx = d;
        }
    }

    sdata[threadIdx.x] = make_float2(thread_max, thread_idx);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float2 other = sdata[threadIdx.x + s];
            if (other.x > sdata[threadIdx.x].x || 
                (other.x == sdata[threadIdx.x].x && other.y < sdata[threadIdx.x].y))
            {
                sdata[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        indices[k] = static_cast<int64_t>(sdata[0].y);
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
    
    const int threads = 256;
    const int blocks = outerSize * innerSize;
    const size_t smem = threads * 2 * sizeof(float);
    
    argmax_aligned_memory_kernel<float><<<blocks, threads, smem>>>(
        reinterpret_cast<const float*>(__align__(16) x_contig.data_ptr()),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );
    
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (aligned)");
}