#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Combined kernel: performs grid-stride iteration, warp-level reduction using shuffle intrinsics,
// and then a shared-memory reduction of warp partial sums.

template <int BLOCK_SIZE>
__global__ void hinge_loss_combined_kernel(const float* __restrict__ predictions,
                                              const float* __restrict__ targets,
                                              float* __restrict__ partialSums,
                                              int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    float localSum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = idx; i < n; i += gridDim.x * BLOCK_SIZE) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        localSum += fmaxf(0.0f, 1.0f - pred * targ);
    }

    // Warp-level reduction using shuffle intrinsics
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(mask, localSum, offset);
    }

    // Each warp's first lane writes its sum to shared memory
    __shared__ float warpSums[BLOCK_SIZE / 32];
    int warpId = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) {
        warpSums[warpId] = localSum;
    }
    __syncthreads();

    // Use the first warp to reduce the warp-level sums
    if (tid < (BLOCK_SIZE / 32)) {
        localSum = warpSums[tid];
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset /= 2) {
            localSum += __shfl_down_sync(mask, localSum, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = localSum;
        }
    }
}

// Host function: selects block size based on problem size and launches the combined kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
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

    // Allocate tensor for block partial sums
    auto partialSums = torch::empty({blocks}, predictions.options());

    // Launch the kernel with the appropriate block size specialization
    switch(block_size) {
        case 32:
            hinge_loss_combined_kernel<32><<<blocks, 32>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n);
            break;
        case 64:
            hinge_loss_combined_kernel<64><<<blocks, 64>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n);
            break;
        case 128:
            hinge_loss_combined_kernel<128><<<blocks, 128>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n);
            break;
        case 256:
            hinge_loss_combined_kernel<256><<<blocks, 256>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n);
            break;
        case 512:
            hinge_loss_combined_kernel<512><<<blocks, 512>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n);
            break;
        default:
            hinge_loss_combined_kernel<256><<<blocks, 256>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                partialSums.data_ptr<float>(),
                n);
            break;
    }

    // Final reduction: sum up block results and compute the mean hinge loss
    return torch::sum(partialSums) / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient Hinge Loss Forward");
}
