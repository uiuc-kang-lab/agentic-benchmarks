#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
// Maximum number of floats that can be stored in constant memory
#define MAX_CONST_INPUT_SIZE 16384

// Declare constant memory for input tensor
__constant__ float d_const_input[MAX_CONST_INPUT_SIZE];

// Kernel using constant memory for input data
__global__ void softmax_kernel_const(float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Each block processes one row; read data from constant memory
    const float* x_row = d_const_input + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    __shared__ float sdata[THREADS_PER_BLOCK]; // shared memory for reductions

    // 1. Compute maximum value in the row
    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = x_row[i];
        max_val = fmaxf(max_val, val);
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Reduction for maximum
    for (int s = stride / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // 2. Compute exponentials and accumulate sum
    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum_val += exp_val;
    }
    sdata[tid] = sum_val;
    __syncthreads();

    // Reduction for sum
    for (int s = stride / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum_val = sdata[0];
    __syncthreads();

    // 3. Normalize the computed exponentials
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / sum_val;
    }
}

// Fallback kernel using global memory (reference implementation) when input tensor is too large for constant memory
__global__ void softmax_kernel_global(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    extern __shared__ float sdata[];

    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = x_row[i];
        max_val = fmaxf(max_val, val);
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = stride / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum_val += exp_val;
    }
    sdata[tid] = sum_val;
    __syncthreads();

    for (int s = stride / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    sum_val = sdata[0];
    __syncthreads();

    for (int i = tid; i < num_features; i += stride) {
        y_row[i] /= sum_val;
    }
}

// Host function to launch the appropriate kernel
void softmax_forward_cuda_const(const float* x, float* y, int batch_size, int num_features) {
    int total_elements = batch_size * num_features;
    int total_bytes = total_elements * sizeof(float);
    
    // Ensure the input tensor fits in constant memory
    TORCH_CHECK(total_elements <= MAX_CONST_INPUT_SIZE, "Input tensor is too large for constant memory kernel");

    // Copy the input tensor to constant memory (device-to-device copy)
    cudaMemcpyToSymbol(d_const_input, x, total_bytes, 0, cudaMemcpyDeviceToDevice);

    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);

    softmax_kernel_const<<<grid_dim, block_dim, shared_mem_size>>>(y, num_features);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda_const: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Main forward function exposed to Python
torch::Tensor forward(torch::Tensor x) {
    // Input validations
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);
    int total_elements = batch_size * num_features;

    auto y = torch::empty_like(x);

    if (total_elements <= MAX_CONST_INPUT_SIZE) {
        // Use constant memory optimized kernel
        softmax_forward_cuda_const(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    } else {
        // Fallback to global memory kernel if input is too large for constant memory
        dim3 grid_dim(batch_size);
        dim3 block_dim(THREADS_PER_BLOCK);
        int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);
        softmax_kernel_global<<<grid_dim, block_dim, shared_mem_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_features);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in softmax_forward_cuda_global: %s\n", cudaGetErrorString(err));
        }
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward with constant memory (CUDA)");
}
