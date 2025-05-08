#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Experiment with block sizes; optimal configuration found: 128 threads per block
#define BLOCK_SIZE 128

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    extern __shared__ float sdata[];

    // Pass 1: Compute maximum value of the row
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        thread_max = max(thread_max, x_row[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Reduction for maximum
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Pass 2: Compute exponentials and partial sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;  // store temporary exponential
        thread_sum += exp_val;
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Reduction for sum
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];

    // Pass 3: Normalize the values
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(BLOCK_SIZE);
    int shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
    
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
