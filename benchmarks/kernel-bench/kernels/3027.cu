#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// CUDA kernel with manually unrolled warp-level reductions
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block handles one row from the batch
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // dynamic shared memory used to store a global scalar (first for max, then for sum)
    extern __shared__ float global_mem[];  // only index 0 is used

    // statically allocated shared memory for warp-level reductions (for max and sum)
    __shared__ float warp_max[THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float warp_sum[THREADS_PER_BLOCK / WARP_SIZE];

    // Step 1: Compute the maximum value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = x_row[i];
        if (val > thread_max) {
            thread_max = val;
        }
    }

    // Unroll the warp-level reduction for maximum using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, thread_max, offset);
        thread_max = fmaxf(thread_max, other);
    }

    int warp_id = tid / WARP_SIZE;
    if ((tid % WARP_SIZE) == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Reduce max values across warps; only the first (stride/WARP_SIZE) threads participate
    if (tid < (stride / WARP_SIZE)) {
        float local_max = warp_max[tid];
        #pragma unroll
        for (int offset = (stride / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, other);
        }
        if (tid == 0) {
            global_mem[0] = local_max;  // store the global max
        }
    }
    __syncthreads();

    float max_val = global_mem[0];

    // Step 2: Compute exponentials and their partial sums
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;  // store temporary exponential values
        thread_sum += exp_val;
    }

    // Unroll the warp-level reduction for sum using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
    }

    if ((tid % WARP_SIZE) == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Reduce sums across warps
    if (tid < (stride / WARP_SIZE)) {
        float local_sum = warp_sum[tid];
        #pragma unroll
        for (int offset = (stride / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        }
        if (tid == 0) {
            global_mem[0] = local_sum;  // store the global sum
        }
    }
    __syncthreads();

    float sum_val = global_mem[0];

    // Step 3: Normalize the outputs
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / sum_val;
    }
}

// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    // Only 1 float is needed in dynamic (global) shared memory
    int dynamic_shared_size = sizeof(float);
    softmax_kernel<<<grid_dim, block_dim, dynamic_shared_size>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ forward function
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

// pybind11 module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
