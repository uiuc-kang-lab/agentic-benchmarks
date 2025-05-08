#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __inline__ float4 vectorized_load(const float* addr, int idx) {
    return __ldg(reinterpret_cast<const float4*>(addr) + idx);
}

__device__ __inline__ void vectorized_store(float* addr, int idx, float4 val) {
    reinterpret_cast<float4*>(addr)[idx] = val;
}

__device__ float compute_sum(const float* row_start, int D, float* shared) {
    float sum = 0.0f;
    const int nVec = D / 4;
    const int rem = D % 4;
    
    // Use shared memory tiling to reduce global memory traffic
    const int TILE_SIZE = 256; // Process data in chunks
    const int num_tiles = (nVec + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * TILE_SIZE;
        const int tile_elements = min(TILE_SIZE, nVec - tile_start);
        
        // Load tile into shared memory
        for (int i = threadIdx.x; i < tile_elements; i += blockDim.x) {
            float4 val = vectorized_load(row_start, tile_start + i);
            shared[i * 4] = val.x;
            shared[i * 4 + 1] = val.y;
            shared[i * 4 + 2] = val.z;
            shared[i * 4 + 3] = val.w;
        }
        __syncthreads();
        
        // Process data from shared memory
        for (int i = threadIdx.x; i < tile_elements * 4; i += blockDim.x) {
            sum += fabsf(shared[i]);
        }
        __syncthreads();
    }

    // Handle remaining elements
    const int vec_elems = nVec * 4;
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        sum += fabsf(__ldg(row_start + vec_elems + j));
    }

    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ void handle_zero_sum(float* shared) {
    if (threadIdx.x == 0 && shared[0] == 0.0f) {
        shared[0] = 1e-12f;
    }
    __syncthreads();
}

__device__ void normalize_row(const float* x_row, float* out_row, float total_sum, int D) {
    const int nVec = D / 4;
    const int rem = D % 4;

    for (int i = threadIdx.x; i < nVec; i += blockDim.x) {
        float4 val = vectorized_load(x_row, i);
        float4 res = {
            val.x / total_sum,
            val.y / total_sum,
            val.z / total_sum,
            val.w / total_sum
        };
        vectorized_store(out_row, i, res);
    }

    const int vec_elems = nVec * 4;
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        out_row[vec_elems + j] = __ldg(x_row + vec_elems + j) / total_sum;
    }
}

__global__ void l1_norm_forward_kernel_opt(const float* __restrict__ x,
                                            float* __restrict__ out,
                                            int N, int D) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const float* x_row = x + row * D;

    float total_sum = compute_sum(x_row, D, sdata);
    handle_zero_sum(sdata);
    total_sum = sdata[0];

    normalize_row(x_row, out + row * D, total_sum, D);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int D = x.size(1);

    const int threads = std::min(1024, D);
    const size_t shared_size = threads * sizeof(float);

    l1_norm_forward_kernel_opt<<<N, threads, shared_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular L1 Normalization (CUDA)");
}