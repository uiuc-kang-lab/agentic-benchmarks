#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// This kernel uses grid-stride loops to evenly distribute the workload for each row (batch element).
// Each block processes one row. Threads loop over features in strides of blockDim.x, ensuring that if
// num_features is less than THREADS_PER_BLOCK, only the needed number of threads are active.
// The kernel performs three phases: reduction to compute the row maximum, computation of exp(x - max) and sum
// (also stored for later normalization), followed by a normalization step. This distribution minimizes
// idle threads and avoids bottlenecks when processing rows with various feature sizes.

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int row = blockIdx.x;  // each block handles one row
    int tid = threadIdx.x;

    extern __shared__ float sdata[];  // shared memory for reduction

    // Phase 1: Compute maximum value in the row using a grid-stride loop
    float local_max = -INFINITY;
    for (int j = tid; j < num_features; j += blockDim.x) {
        float val = x[row * num_features + j];
        local_max = fmaxf(local_max, val);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduce to get the maximum value of the row
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Phase 2: Compute sum of exp(x - max) and write exp values into y
    float local_sum = 0.0f;
    for (int j = tid; j < num_features; j += blockDim.x) {
        float exp_val = __expf(x[row * num_features + j] - max_val);
        y[row * num_features + j] = exp_val;
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce to get the sum of exponentials for the row
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    __syncthreads();

    // Phase 3: Normalize the softmax output
    for (int j = tid; j < num_features; j += blockDim.x) {
        y[row * num_features + j] /= sum_val;
    }
}


// Host function to launch the softmax kernel. It selects the number of threads based on num_features
// to evenly distribute the workload and avoid underutilization when num_features is small.
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    // Choose the number of threads as the minimum between THREADS_PER_BLOCK and num_features
    int threads = (num_features < THREADS_PER_BLOCK) ? num_features : THREADS_PER_BLOCK;
    dim3 grid(batch_size);
    dim3 block(threads);
    size_t shared_mem_size = threads * sizeof(float);
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ interface for PyTorch

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    int batch_size = x.size(0);
    int num_features = x.size(1);
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-stride Softmax forward (CUDA)");
}
