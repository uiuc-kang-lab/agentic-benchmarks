#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimizing kernel by leveraging shared memory to minimize global memory access latency

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* anchor,
    const scalar_t* positive,
    const scalar_t* negative,
    scalar_t* output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    // Shared memory for distances to reduce global memory access
    extern __shared__ char shared_mem[];
    scalar_t* shared_pos = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* shared_neg = shared_pos + blockDim.x;
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int stride = blockDim.x;
    scalar_t dist_pos = 0;
    scalar_t dist_neg = 0;
    
    // Load data into shared memory
    for (int idx = tid; idx < feat_size; idx += stride) {
        const int offset = batch_idx * feat_size + idx;
        const scalar_t a = anchor[offset];
        const scalar_t p = positive[offset];
        const scalar_t n = negative[offset];

        // Distance computations
        const scalar_t d_pos = a - p;
        const scalar_t d_neg = a - n;
        dist_pos += d_pos * d_pos;
        dist_neg += d_neg * d_neg;
    }
    
    // Store intermediate distances into shared memory
    shared_pos[tid] = dist_pos;
    shared_neg[tid] = dist_neg;
    __syncthreads();
    
    // Reduce within block using shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_pos[tid] += shared_pos[tid + stride];
            shared_neg[tid] += shared_neg[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result to output for thread 0 of each block
    if (tid == 0) {
        const scalar_t loss = max(scalar_t(0.0), sqrt(shared_pos[0]) - sqrt(shared_neg[0]) + margin);
        output[batch_idx] = loss;
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {
    
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    
    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);
    auto output = torch::zeros({batch_size}, anchor.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem_size = 2 * threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            batch_size,
            feat_size);
    }));
    
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA)");
}