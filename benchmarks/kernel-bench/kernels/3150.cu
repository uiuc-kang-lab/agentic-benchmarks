#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// This kernel makes use of 3D blocking to allow for more flexible distribution of threads.
// Allows us to handle larger dimensions more efficiently by dedicating layers of the 3D grid to different parts of the tensor.

__global__ void log_softmax_3d_blocking(const float* input, float* output, int dim_size, int batch_size) {
    // Use 3D grid and block to handle more data-intensive tasks
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // Depth index

    // Check if within bounds
    if (col >= dim_size || row >= batch_size) return;

    extern __shared__ float sdata[];
    int idx = col * batch_size + row;
    float max_val = -std::numeric_limits<float>::infinity();

    // Phase 1: Compute maximum value per block-row
    for (int i = z; i < dim_size; i += blockDim.z * gridDim.z) {
        max_val = max(max_val, input[i * batch_size + row]);
    }
    sdata[threadIdx.z] = max_val;
    __syncthreads();

    // Reduction to find maximum value within block (assuming blockDim is power of 2 up to 1024)
    for (int stride = blockDim.z / 2; stride > 0; stride >>= 1) {
        if (threadIdx.z < stride) {
            sdata[threadIdx.z] = max(sdata[threadIdx.z], sdata[threadIdx.z + stride]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Phase 2: Compute sum of exp(x - max) with shared memory
    float local_sum = 0.0f;
    for (int i = z; i < dim_size; i += blockDim.z * gridDim.z) {
        float exp_val = exp(input[i * batch_size + row] - max_val);
        local_sum += exp_val;
        output[i * batch_size + row] = exp_val;
    }
    sdata[threadIdx.z] = local_sum;
    __syncthreads();

    // Reduction to compute total sum
    for (int stride = blockDim.z / 2; stride > 0; stride >>= 1) {
        if (threadIdx.z < stride) {
            sdata[threadIdx.z] += sdata[threadIdx.z + stride];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    float log_sum = log(sum_val);
    __syncthreads();

    // Phase 3: Compute final output
    for (int i = z; i < dim_size; i += blockDim.z * gridDim.z) {
        output[i * batch_size + row] = output[i * batch_size + row] - log_sum;
    }
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input so the target dim is last
    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            permute_dims.push_back(i);
        }
    }
    permute_dims.push_back(dim);
    input = input.permute(permute_dims).contiguous();

    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    // Experiment with block size settings
    dim3 threads(1, 32, 32);  // 1 thread for each depth of the dimension being softmaxed
    dim3 blocks(batch_size, (dim_size + threads.y - 1) / threads.y);
    
    // Assuming use of float for input as the max shared memory requirement
    size_t shared_mem_size = threads.z * sizeof(float);

    log_softmax_3d_blocking<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size,
        batch_size);

    // Inverse permutation
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA) with 3D blocking");
}