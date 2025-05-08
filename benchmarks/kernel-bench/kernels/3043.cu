#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define MAX_FEATURES 2048

__global__ void softmax_kernel(const float* __restrict__ x, 
                             float* __restrict__ y, 
                             int batch_size,
                             int num_features) {
    extern __shared__ float smem[];
    float* warp_max = smem;
    float* warp_sum = &smem[THREADS_PER_BLOCK/WARP_SIZE];
    
    // Process multiple rows per block for better utilization
    for (int row = blockIdx.x; row < batch_size; row += gridDim.x) {
        const float* row_input = x + row * num_features;
        float* row_output = y + row * num_features;
        
        int tid = threadIdx.x;
        int warp_id = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;
        
        // Calculate thread block's workload
        int elements_per_thread = (num_features + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int start_idx = tid * elements_per_thread;
        int end_idx = min(start_idx + elements_per_thread, num_features);
        
        // Find max value
        float thread_max = -INFINITY;
        #pragma unroll 4
        for (int i = start_idx; i < end_idx; i++) {
            thread_max = max(thread_max, row_input[i]);
        }
        
        // Warp-level reduction for max
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
            thread_max = max(thread_max, other_max);
        }
        
        // Store warp results
        if (lane == 0) {
            warp_max[warp_id] = thread_max;
        }
        __syncthreads();
        
        // Final reduction for max across warps
        if (tid < (THREADS_PER_BLOCK/WARP_SIZE)) {
            float warp_maximum = warp_max[tid];
            #pragma unroll
            for (int i = tid + 1; i < (THREADS_PER_BLOCK/WARP_SIZE); i++) {
                warp_maximum = max(warp_maximum, warp_max[i]);
            }
            warp_max[tid] = warp_maximum;
        }
        __syncthreads();
        
        float max_val = warp_max[0];
        
        // Compute exponentials and sum
        float thread_sum = 0.0f;
        #pragma unroll 4
        for (int i = start_idx; i < end_idx; i++) {
            float val = __expf(row_input[i] - max_val);
            row_output[i] = val;  // Store intermediate result
            thread_sum += val;
        }
        
        // Warp-level reduction for sum
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (lane == 0) {
            warp_sum[warp_id] = thread_sum;
        }
        __syncthreads();
        
        // Final reduction for sum across warps
        if (tid < (THREADS_PER_BLOCK/WARP_SIZE)) {
            float warp_total = warp_sum[tid];
            #pragma unroll
            for (int i = tid + 1; i < (THREADS_PER_BLOCK/WARP_SIZE); i++) {
                warp_total += warp_sum[i];
            }
            warp_sum[tid] = warp_total;
        }
        __syncthreads();
        
        float sum_val = warp_sum[0];
        
        // Normalize the exponentials
        #pragma unroll 4
        for (int i = start_idx; i < end_idx; i++) {
            row_output[i] /= sum_val;
        }
        __syncthreads();
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    // Calculate optimal grid size based on SM count
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    
    // Use multiple of SM count for grid size, but cap it at batch_size
    int optimal_blocks = min(batch_size, sm_count * 4);
    
    dim3 grid(optimal_blocks);
    dim3 block(THREADS_PER_BLOCK);
    
    // Shared memory size calculation
    size_t shared_mem_size = (2 * THREADS_PER_BLOCK/WARP_SIZE) * sizeof(float);
    
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, batch_size, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    
    auto y = torch::empty_like(x);
    softmax_forward_cuda(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x.size(0),
        x.size(1)
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}