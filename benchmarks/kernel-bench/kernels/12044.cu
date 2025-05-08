#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Templated kernel that performs hinge loss computation with reduction
// using a grid-stride loop and shared memory reduction.

template <int BLOCK_SIZE>
__global__ void hinge_loss_tuned_kernel(const float* __restrict__ predictions,
                                         const float* __restrict__ targets,
                                         float* __restrict__ partialSums,
                                         int n) {
    __shared__ float shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    float localSum = 0.0f;
    
    // Grid-stride loop: allow each thread to process multiple elements
    for (int i = idx; i < n; i += gridDim.x * BLOCK_SIZE) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        localSum += fmaxf(0.0f, 1.0f - pred * targ);
    }
    
    shared_data[tid] = localSum;
    __syncthreads();
    
    // In-block reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result of this block to global memory
    if (tid == 0) {
        partialSums[blockIdx.x] = shared_data[0];
    }
}

// The forward function selects the block size based on the problem size
// and launches the appropriate templated kernel specialization.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    
    int n = predictions.numel();

    // Choose block size based on problem size to experiment with different configurations
    int block_size;
    if (n < 512) {
        block_size = 32;
    } else if (n < 4096) {
        block_size = 64;
    } else if (n < 100000) {
        block_size = 128;
    } else if (n < 10000000) {
        block_size = 256;
    } else {
        block_size = 512;
    }
    
    int blocks = (n + block_size - 1) / block_size;
    auto partialSums = torch::empty({blocks}, predictions.options());
    
    // Launch the kernel with the chosen block size
    switch(block_size) {
        case 32:
            hinge_loss_tuned_kernel<32><<<blocks, 32>>>(predictions.data_ptr<float>(),
                                                          targets.data_ptr<float>(),
                                                          partialSums.data_ptr<float>(),
                                                          n);
            break;
        case 64:
            hinge_loss_tuned_kernel<64><<<blocks, 64>>>(predictions.data_ptr<float>(),
                                                          targets.data_ptr<float>(),
                                                          partialSums.data_ptr<float>(),
                                                          n);
            break;
        case 128:
            hinge_loss_tuned_kernel<128><<<blocks, 128>>>(predictions.data_ptr<float>(),
                                                          targets.data_ptr<float>(),
                                                          partialSums.data_ptr<float>(),
                                                          n);
            break;
        case 256:
            hinge_loss_tuned_kernel<256><<<blocks, 256>>>(predictions.data_ptr<float>(),
                                                          targets.data_ptr<float>(),
                                                          partialSums.data_ptr<float>(),
                                                          n);
            break;
        case 512:
            hinge_loss_tuned_kernel<512><<<blocks, 512>>>(predictions.data_ptr<float>(),
                                                          targets.data_ptr<float>(),
                                                          partialSums.data_ptr<float>(),
                                                          n);
            break;
        default:
            hinge_loss_tuned_kernel<256><<<blocks, 256>>>(predictions.data_ptr<float>(),
                                                          targets.data_ptr<float>(),
                                                          partialSums.data_ptr<float>(),
                                                          n);
            break;
    }
    
    // Final reduction: compute the mean hinge loss on the GPU
    return torch::sum(partialSums) / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tuned Hinge Loss Forward");
}
