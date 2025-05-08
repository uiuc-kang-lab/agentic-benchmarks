#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename T>
__device__ __forceinline__ T myMin(T a, T b) {
    return a < b ? a : b;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceMin(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(0xffffffff, val, offset);
        val = (temp < val) ? temp : val;
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceMin(scalar_t val) {
    __shared__ scalar_t shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMin(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : std::numeric_limits<scalar_t>::max();
    if (wid == 0) val = warpReduceMin(val);

    return val;
}

template <typename scalar_t>
__global__ void aligned_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
    
    int idx = blockIdx.x;
    if (idx >= outer * inner) return;

    int outer_idx = idx / inner;
    int inner_idx = idx % inner;
    
    const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    
    if (std::is_same<scalar_t, float>::value && inner >= 4 && 
        (reinterpret_cast<uintptr_t>(in_ptr) & 15) == 0 && r >= 4) {
        const float4* vec_ptr = reinterpret_cast<const float4*>(in_ptr);
        int vec_elements = r / 4;
        
        for (int j = threadIdx.x; j < vec_elements; j += blockDim.x) {
            float4 vec_val = __ldg(vec_ptr + j * (inner/4));
            local_min = min(local_min, min(min(vec_val.x, vec_val.y), 
                                        min(vec_val.z, vec_val.w)));
        }
        
        for (int j = vec_elements * 4 + threadIdx.x; j < r; j += blockDim.x) {
            scalar_t val = __ldg(in_ptr + j * inner);
            local_min = min(local_min, val);
        }
    } else {
        for (int j = threadIdx.x; j < r; j += blockDim.x) {
            scalar_t val = __ldg(in_ptr + j * inner);
            local_min = min(local_min, val);
        }
    }
    
    local_min = blockReduceMin(local_min);
    
    if (threadIdx.x == 0) {
        output[idx] = local_min;
    }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= input.size(i);
    }
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner *= input.size(i);
    }

    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }

    auto output = torch::empty(output_shape, input.options());
    int total = outer * inner;

    int threads = 256;
    if (r < 256) threads = ((r + 31) / 32) * 32;
    int blocks = total;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "aligned_min_reduce_cuda", ([&] {
        aligned_min_reduction_kernel<scalar_t><<<blocks, threads, 0, 
            c10::cuda::getCurrentCUDAStream().stream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer,
            r,
            inner);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aligned memory min reduction over a dimension (CUDA)");
}