#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that distributes workload evenly across threads and blocks using a grid-stride loop
// and performs block-level reduction to minimize atomic contention

__global__ void distributed_hinge_loss_kernel(const float* __restrict__ predictions,
                                                const float* __restrict__ targets,
                                                float* global_sum,
                                                int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local_sum = 0.0f;
    // Grid-stride loop to distribute work evenly
    for (int i = idx; i < n; i += stride) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        local_sum += fmaxf(0.0f, 1.0f - pred * targ);
    }
    
    // Store the partial sum in shared memory
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduce within the block using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Use atomic add to accumulate the block's sum into the global sum
    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// PyTorch binding function

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    auto global_sum = torch::zeros({1}, predictions.options());

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Launch kernel with dynamically allocated shared memory for block reduction
    distributed_hinge_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum.data_ptr<float>(),
        n
    );

    // Compute and return the mean hinge loss
    return global_sum / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Distributed Hinge Loss Forward Kernel");
}
