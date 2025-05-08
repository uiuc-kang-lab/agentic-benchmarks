#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Templated softmax kernel that allows tuning the block size at compile time
template <int BLOCK_SIZE>
__global__ void softmax_kernel_template(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    const int blockSize = BLOCK_SIZE;

    // Pointers to the start of the current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Allocate shared memory: first blockSize for max reduction, next blockSize for sum reduction
    extern __shared__ float shared_mem[];
    float* max_shared = shared_mem;
    float* sum_shared = shared_mem + blockSize;

    // Step 1: Compute the maximum value in the row using a stride loop
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockSize) {
        thread_max = fmaxf(thread_max, x_row[i]);
    }
    max_shared[tid] = thread_max;
    __syncthreads();

    // Reduction to compute the max value
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + s]);
        }
        __syncthreads();
    }

    float row_max = max_shared[0];
    __syncthreads();

    // Step 2: Compute exponentials and accumulate partial sums
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockSize) {
        float exp_val = __expf(x_row[i] - row_max);
        y_row[i] = exp_val; // store intermediate result
        thread_sum += exp_val;
    }
    sum_shared[tid] = thread_sum;
    __syncthreads();

    // Reduction to compute the sum of exponentials
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
        }
        __syncthreads();
    }

    float sum_val = sum_shared[0];
    __syncthreads();

    // Step 3: Normalize the results
    for (int i = tid; i < num_features; i += blockSize) {
        y_row[i] /= sum_val;
    }
}

// Host function to launch the kernel with a tunable block size
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features, int block_size) {
    dim3 grid_dim(batch_size);
    int shared_mem_size = sizeof(float) * block_size * 2; // for max and sum arrays

    switch(block_size) {
        case 32: {
            dim3 block_dim(32);
            softmax_kernel_template<32><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 64: {
            dim3 block_dim(64);
            softmax_kernel_template<64><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 128: {
            dim3 block_dim(128);
            softmax_kernel_template<128><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 256: {
            dim3 block_dim(256);
            softmax_kernel_template<256><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 512: {
            dim3 block_dim(512);
            softmax_kernel_template<512><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        default: {
            // Default to 256 if an unsupported block size is provided
            dim3 block_dim(256);
            softmax_kernel_template<256><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
    }
}

// C++ forward function exposed to PyTorch
// Added optional parameter 'block_size' to allow experimentation with different configurations
torch::Tensor forward(torch::Tensor x, int block_size = 256) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features, block_size);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA) with tunable block size",
          py::arg("x"), py::arg("block_size") = 256);
}
