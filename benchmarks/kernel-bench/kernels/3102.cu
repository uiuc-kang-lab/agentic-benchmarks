#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>


// Define the number of threads per block and the maximum elements that can be stored in constant memory
#define THREADS_PER_BLOCK 256
#define MAX_CONST_SIZE 16384  // Maximum number of float elements (64KB) in constant memory

// Constant memory declaration for read-only input data
__constant__ float d_const_x[MAX_CONST_SIZE];

// CUDA kernel that uses constant memory for input tensor
__global__ void softmax_kernel_const(float* __restrict__ y, int num_features) {
    // Each block processes one row
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Get pointers to the current row in constant memory and output
    const float* x_row = d_const_x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Allocate shared memory for reduction
    extern __shared__ float sdata[];

    // Step 1: Compute the maximum value in the row
    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = x_row[i];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Each thread writes its local maximum to shared memory
    if (tid < stride) {
        sdata[tid] = max_val;
    }
    __syncthreads();

    // Reduction to get the row-wise maximum
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Step 2: Compute exponentials and their sum
    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;  // Temporarily store the exponentials in the output
        sum_val += exp_val;
    }

    // Reduction to compute the sum of exponentials
    if (tid < stride) {
        sdata[tid] = sum_val;
    }
    __syncthreads();
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum_val = sdata[0];
    __syncthreads();

    // Step 3: Normalize the exponentials to obtain softmax probabilities
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / sum_val;
    }
}

// Host function to launch the CUDA kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    int total_elements = batch_size * num_features;
    int size_in_bytes = total_elements * sizeof(float);

    // Ensure the total number of elements fits into our constant memory buffer
    TORCH_CHECK(total_elements <= MAX_CONST_SIZE, "Input tensor size exceeds constant memory limit.");

    // Copy the input data to constant memory; using DeviceToDevice copy because x is already a CUDA pointer
    cudaError_t cpyErr = cudaMemcpyToSymbol(d_const_x, x, size_in_bytes, 0, cudaMemcpyDeviceToDevice);
    if (cpyErr != cudaSuccess) {
        printf("Error copying to constant memory: %s\n", cudaGetErrorString(cpyErr));
        return;
    }

    // Configure grid and block dimensions
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);

    // Launch the kernel
    softmax_kernel_const<<<grid_dim, block_dim, shared_mem_size>>>(y, num_features);

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor x) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);
    int total_elements = batch_size * num_features;

    // Ensure the tensor fits in constant memory
    TORCH_CHECK(total_elements <= MAX_CONST_SIZE, "Input tensor size exceeds constant memory limit.");

    // Allocate output tensor
    auto y = torch::empty_like(x);

    // Launch CUDA kernel
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);

    return y;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA) using constant memory optimization");
}
