#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

const int NUM_STREAMS = 4;  // Number of concurrent streams

__global__ void l1_norm_forward_kernel_stream(const float* __restrict__ x,
                                            float* __restrict__ out,
                                            int N,
                                            int D,
                                            int chunk_start,
                                            int chunk_size) {
    extern __shared__ float sdata[];
    int row = blockIdx.x + chunk_start;
    if (row >= N || row >= (chunk_start + chunk_size)) return;
    
    int row_start = row * D;
    float sum = 0.0f;

    // Process elements in groups of 4 (128-bit)
    int nVec = D / 4;
    int rem = D % 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x + row_start);

    // Vectorized processing
    for (int i = threadIdx.x; i < nVec; i += blockDim.x) {
        float4 val = __ldg(x_vec + i);
        sum += fabsf(val.x) + fabsf(val.y) + fabsf(val.z) + fabsf(val.w);
    }

    // Handle remaining elements
    int vec_elems = nVec * 4;
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        float val = __ldg(x + row_start + vec_elems + j);
        sum += fabsf(val);
    }

    // Reduction in shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float total_sum = sdata[0];
    if (threadIdx.x == 0 && total_sum == 0.0f) {
        total_sum = 1e-12f;
        sdata[0] = total_sum;
    }
    __syncthreads();
    total_sum = sdata[0];

    // Normalized output with vectorized writes
    float4* out_vec = reinterpret_cast<float4*>(out + row_start);
    for (int i = threadIdx.x; i < nVec; i += blockDim.x) {
        float4 val = __ldg(x_vec + i);
        float4 res;
        res.x = val.x / total_sum;
        res.y = val.y / total_sum;
        res.z = val.z / total_sum;
        res.w = val.w / total_sum;
        out_vec[i] = res;
    }

    // Handle remaining elements
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        int idx = row_start + nVec * 4 + j;
        out[idx] = x[idx] / total_sum;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threads = std::min<int>(1024, D);
    int shared_mem_size = threads * sizeof(float);
    
    // Calculate chunk size for each stream
    int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_start = i * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - chunk_start);
        
        if (current_chunk_size > 0) {
            l1_norm_forward_kernel_stream<<<current_chunk_size, threads, shared_mem_size, streams[i]>>>(
                x.data_ptr<float>(),
                out.data_ptr<float>(),
                N,
                D,
                chunk_start,
                current_chunk_size
            );
        }
    }

    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined L1 Normalization forward pass (CUDA)");
}