#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void triplet_margin_loss_opt_kernel(
    const scalar_t* anchor,
    const scalar_t* positive,
    const scalar_t* negative,
    scalar_t* output,
    const float margin,
    const int batch_size,
    const int feat_size) {
    
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_bytes[];
    scalar_t* shared_pos = reinterpret_cast<scalar_t*>(shared_bytes);
    scalar_t* shared_neg = shared_pos + blockDim.x;
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    scalar_t dist_pos = 0;
    scalar_t dist_neg = 0;

    for (int i = tid; i < feat_size; i += blockDim.x) {
        const int idx = batch_idx * feat_size + i;
        scalar_t a = anchor[idx];
        scalar_t p = positive[idx];
        scalar_t n = negative[idx];
        
        dist_pos += (a - p) * (a - p);
        dist_neg += (a - n) * (a - n);
    }

    shared_mem[tid] = dist_pos;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    dist_pos = shared_mem[0];
    shared_mem[tid] = dist_neg;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    dist_neg = shared_mem[0];

    if (tid == 0) {
        scalar_t loss = max(scalar_t(0), sqrt(dist_pos) - sqrt(dist_neg) + margin);
        output[batch_idx] = loss;
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {
    
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be CUDA tensor");
    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);
    auto output = torch::zeros({batch_size}, anchor.options());

    const int threads = 256;
    const int shared_mem = 2 * threads * anchor.element_size();
    triplet_margin_loss_opt_kernel<scalar_t>
        <<<batch_size, threads, shared_mem>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            margin,
            batch_size,
            feat_size
        );

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Optimized TripletMarginLoss (CUDA)");
}