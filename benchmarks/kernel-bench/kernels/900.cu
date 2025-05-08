#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM 2

__device__ __forceinline__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void balanced_matmul_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int M, const int K, const int N) {
    // Calculate optimal work distribution
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int num_blocks = gridDim.x * gridDim.y;
    const int num_threads = blockDim.x * blockDim.y;
    
    // Calculate total elements and work per thread
    const int total_elements = M * N;
    const int elements_per_block = (total_elements + num_blocks - 1) / num_blocks;
    const int elements_per_thread = (elements_per_block + num_threads - 1) / num_threads;
    
    // Calculate starting position for this thread
    const int start_idx = (bid * elements_per_block + tid * elements_per_thread);
    
    extern __shared__ float sdata[];
    
    // Process multiple elements per thread
    for (int idx = start_idx; idx < min(start_idx + elements_per_thread, total_elements); idx++) {
        const int row = idx / N;
        const int col = idx % N;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            
            // Compute dot product with vectorized loads
            #pragma unroll 4
            for (int k = 0; k < K; k += 4) {
                float4 a_vec, b_vec;
                if (k + 3 < K) {
                    // Load 4 elements at once
                    float4 a_load = *reinterpret_cast<const float4*>(&A[row * K + k]);
                    float4 b_load = *reinterpret_cast<const float4*>(&B[k * N + col]);
                    sum += a_load.x * b_load.x + a_load.y * b_load.y + 
                          a_load.z * b_load.z + a_load.w * b_load.w;
                } else {
                    // Handle remaining elements
                    for (int kk = k; kk < K; kk++) {
                        sum += A[row * K + kk] * B[kk * N + col];
                    }
                }
            }
            
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Dynamic threshold based on matrix dimensions
    const bool useCustomKernel = (M * N * K <= 512 * 512 * 512);

    if (useCustomKernel) {
        // Calculate optimal block and grid dimensions
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, deviceId);

        const int num_sms = props.multiProcessorCount;
        const int max_threads_per_sm = props.maxThreadsPerMultiProcessor;
        
        // Calculate optimal thread block size
        dim3 block_dim;
        if (M * N <= MAX_THREADS_PER_BLOCK) {
            block_dim = dim3(32, (M * N + 31) / 32);
        } else {
            block_dim = dim3(16, 16);
        }

        // Calculate grid dimensions to ensure even distribution across SMs
        const int target_blocks_per_sm = max_threads_per_sm / (block_dim.x * block_dim.y);
        const int num_blocks = num_sms * target_blocks_per_sm;
        
        dim3 grid_dim((int)ceil(sqrt(num_blocks)), (int)ceil(sqrt(num_blocks)));

        const int shared_mem_size = sizeof(float) * block_dim.x * block_dim.y;
        
        balanced_matmul_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    } else {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // Use cuBLAS for larger matrices
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, M, K, &alpha,
                   B.data_ptr<float>(), N,
                   A.data_ptr<float>(), K,
                   &beta, C.data_ptr<float>(), N);
        
        cublasDestroy(handle);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Balanced Matrix Multiplication (CUDA)");
}