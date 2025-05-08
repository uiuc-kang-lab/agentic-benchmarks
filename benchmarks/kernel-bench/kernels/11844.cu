#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the KL divergence using vectorized loads for 128-bit aligned accesses
// and uses __ldg() to optimize read-only accesses to global memory.

__global__ void kldiv_vect_ldg_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const int vecN) {

    // Total number of threads in the grid
    const int total_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Cast the pointers to float4 for 128-bit (16-byte) vectorized loads
    const float4* logPred4 = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    // Process the vectorized portion: each float4 contains 4 floats
    for (int i = tid; i < vecN; i += total_threads) {
        // Using __ldg to load the data into registers with read-only cache
        float4 lp = __ldg(&logPred4[i]);
        float4 tp = __ldg(&targ4[i]);
        sum += expf(lp.x) - tp.x * lp.x;
        sum += expf(lp.y) - tp.y * lp.y;
        sum += expf(lp.z) - tp.z * lp.z;
        sum += expf(lp.w) - tp.w * lp.w;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp's leader writes its partial sum to shared memory
    __shared__ float warpSums[64]; // enough for up to 2048 threads per block (2048/32 = 64 warps)
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // First warp reduces the partial sums from all warps in this block
    if (threadIdx.x < (blockDim.x >> 5)) {
        float blockSum = warpSums[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            blockSum += __shfl_down_sync(mask, blockSum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, blockSum);
        }
    }

    // Process remainder elements (if n is not a multiple of 4)
    // Only one thread (block 0, thread 0) handles the tail to avoid redundant work
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float remSum = 0.0f;
        int start = vecN * 4; // index from which remainder elements start
        for (int i = start; i < n; i++) {
            float lp = __ldg(log_predictions + i);
            float tp = __ldg(targets + i);
            remSum += expf(lp) - tp * lp;
        }
        atomicAdd(output, remSum);
    }
}

// CUDA function exposed to PyTorch

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Calculate the number of vectorized loads (each load processes 4 elements)
    const int vecN = n / 4;

    // Configure kernel launch parameters
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;  

    kldiv_vect_ldg_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        vecN
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with vectorized __ldg loads");
}
