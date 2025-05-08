#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define VECTOR_LOAD_SIZE 4

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Pointers to current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    // Register-based accumulators
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    // Step 1: Find max using vectorized loads where possible
    int vector_elements = (num_features / VECTOR_LOAD_SIZE) * VECTOR_LOAD_SIZE;
    
    // Vectorized max finding
    for (int i = tid * VECTOR_LOAD_SIZE; i < vector_elements; i += blockDim.x * VECTOR_LOAD_SIZE) {
        float4 vals = reinterpret_cast<const float4*>(x_row)[i/VECTOR_LOAD_SIZE];
        thread_max = max(thread_max, max(max(vals.x, vals.y), max(vals.z, vals.w)));
    }
    
    // Handle remaining elements
    for (int i = vector_elements + tid; i < num_features; i += blockDim.x) {
        thread_max = max(thread_max, x_row[i]);
    }
    
    // Reduce max within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
    }
    
    // First thread in each warp writes to shared memory
    __shared__ float warp_max[8];  // Assuming <= 8 warps per block
    if ((tid & 31) == 0) {
        warp_max[tid >> 5] = thread_max;
    }
    __syncthreads();
    
    // First warp finds final max
    if (tid < 8) {
        thread_max = warp_max[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            thread_max = max(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
        }
    }
    float max_val = __shfl_sync(0xffffffff, thread_max, 0);
    
    // Step 2: Compute exponentials and sum
    thread_sum = 0.0f;
    
    // Vectorized exp computation
    float4 exp_vals;
    for (int i = tid * VECTOR_LOAD_SIZE; i < vector_elements; i += blockDim.x * VECTOR_LOAD_SIZE) {
        float4 vals = reinterpret_cast<const float4*>(x_row)[i/VECTOR_LOAD_SIZE];
        exp_vals.x = __expf(vals.x - max_val);
        exp_vals.y = __expf(vals.y - max_val);
        exp_vals.z = __expf(vals.z - max_val);
        exp_vals.w = __expf(vals.w - max_val);
        
        reinterpret_cast<float4*>(y_row)[i/VECTOR_LOAD_SIZE] = exp_vals;
        thread_sum += exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
    }
    
    // Handle remaining elements
    for (int i = vector_elements + tid; i < num_features; i += blockDim.x) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        thread_sum += exp_val;
    }
    
    // Reduce sum within warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp updates global sum atomically
    if ((tid & 31) == 0) {
        atomicAdd(&warp_max[0], thread_sum);  // Reuse warp_max[0] for sum
    }
    __syncthreads();
    
    float sum_val = warp_max[0];
    
    // Step 3: Normalize with vectorized operations
    for (int i = tid * VECTOR_LOAD_SIZE; i < vector_elements; i += blockDim.x * VECTOR_LOAD_SIZE) {
        float4 vals = reinterpret_cast<float4*>(y_row)[i/VECTOR_LOAD_SIZE];
        vals.x /= sum_val;
        vals.y /= sum_val;
        vals.z /= sum_val;
        vals.w /= sum_val;
        reinterpret_cast<float4*>(y_row)[i/VECTOR_LOAD_SIZE] = vals;
    }
    
    // Handle remaining elements
    for (int i = vector_elements + tid; i < num_features; i += blockDim.x) {
        y_row[i] /= sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    
    softmax_kernel<<<grid_dim, block_dim>>>(x, y, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");
    
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}