#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that uses shared memory for intra-block reduction. Each block handles one batch sample.

template <typename scalar_t>
__global__ void shared_triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int feat_size) {

    // Each block processes one sample in the batch
    const int batch_idx = blockIdx.x;
    const int offset = batch_idx * feat_size;

    // Each thread computes a partial sum over its assigned feature indices
    scalar_t sum_pos = static_cast<scalar_t>(0);
    scalar_t sum_neg = static_cast<scalar_t>(0);

    for (int i = threadIdx.x; i < feat_size; i += blockDim.x) {
        const scalar_t a = anchor[offset + i];
        const scalar_t p = positive[offset + i];
        const scalar_t n = negative[offset + i];
        scalar_t diff_pos = a - p;
        scalar_t diff_neg = a - n;
        sum_pos += diff_pos * diff_pos;
        sum_neg += diff_neg * diff_neg;
    }

    // Allocate shared memory for reduction; we need room for two arrays of size blockDim.x
    extern __shared__ char shared_mem[];
    scalar_t* sdata_pos = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* sdata_neg = sdata_pos + blockDim.x;

    sdata_pos[threadIdx.x] = sum_pos;
    sdata_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_pos[threadIdx.x] += sdata_pos[threadIdx.x + s];
            sdata_neg[threadIdx.x] += sdata_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the final loss value for the sample
    if (threadIdx.x == 0) {
        scalar_t sqrt_pos = sqrt(sdata_pos[0]);
        scalar_t sqrt_neg = sqrt(sdata_neg[0]);
        scalar_t loss = max(static_cast<scalar_t>(0), sqrt_pos - sqrt_neg + margin);
        output[batch_idx] = loss;
    }
}


// Host function that sets up and launches the kernel
torch::Tensor shared_triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    // Allocate output tensor (one loss per sample)
    auto output = torch::zeros({batch_size}, anchor.options());

    // Launch configuration: one block per sample in the batch
    const int threads = 256;
    dim3 blocks(batch_size);
    
    // Compute shared memory size needed: two arrays, each of size 'threads'
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "shared_triplet_margin_loss_kernel", ([&] {
        const int shared_mem_bytes = 2 * threads * sizeof(scalar_t);
        shared_triplet_margin_loss_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            feat_size);
    }));

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_triplet_margin_loss_cuda, "Triplet margin loss forward with shared memory (CUDA)");
}
