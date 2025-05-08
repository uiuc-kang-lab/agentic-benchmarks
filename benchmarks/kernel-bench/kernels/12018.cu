#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Redesigned kernel: one block per batch sample and reduction over feature dimensions

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* anchor,
    const scalar_t* positive,
    const scalar_t* negative,
    scalar_t* output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    int sample = blockIdx.x;  // each block handles one sample
    int tid = threadIdx.x;
    
    scalar_t sum_pos = 0;
    scalar_t sum_neg = 0;

    // Loop over features; each thread processes multiple feature elements
    for (int i = tid; i < feat_size; i += blockDim.x) {
        int idx = sample * feat_size + i;
        scalar_t a = anchor[idx];
        scalar_t p = positive[idx];
        scalar_t n = negative[idx];
        scalar_t d_pos = a - p;
        scalar_t d_neg = a - n;
        sum_pos += d_pos * d_pos;
        sum_neg += d_neg * d_neg;
    }

    // Intra-warp reduction using shuffle instructions
    unsigned mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum_pos += __shfl_down_sync(mask, sum_pos, offset);
        sum_neg += __shfl_down_sync(mask, sum_neg, offset);
    }

    // Shared memory reduction across warps
    __shared__ scalar_t shared_pos[256];
    __shared__ scalar_t shared_neg[256];
    int warp_id = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0) {
        shared_pos[warp_id] = sum_pos;
        shared_neg[warp_id] = sum_neg;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        sum_pos = (threadIdx.x < num_warps) ? shared_pos[threadIdx.x] : 0;
        sum_neg = (threadIdx.x < num_warps) ? shared_neg[threadIdx.x] : 0;
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            sum_pos += __shfl_down_sync(mask, sum_pos, offset);
            sum_neg += __shfl_down_sync(mask, sum_neg, offset);
        }
        if (threadIdx.x == 0) {
            scalar_t diff = sqrt(sum_pos) - sqrt(sum_neg) + margin;
            output[sample] = diff > 0 ? diff : static_cast<scalar_t>(0);
        }
    }
}

// Host function to launch the kernel

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
    
    // Launch one block per batch sample
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads>>>(
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
