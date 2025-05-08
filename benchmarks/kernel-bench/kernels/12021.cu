#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Partial reduction using a 2D grid that distributes work evenly
// across the feature dimension for each sample. Each block handles a contiguous
// chunk of features and atomically adds its partial sum to a global accumulator per sample.

template <typename scalar_t>
__global__ void triplet_margin_loss_partial_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ sum_pos_global,
    scalar_t* __restrict__ sum_neg_global,
    const int feat_size,
    const int blocksPerSample) {

    // 'sample' is the batch index and 'blockIdx.y' is the block's index within the sample
    int sample = blockIdx.x;
    int block_idx = blockIdx.y;
    int base = sample * feat_size;

    // Each block handles a subset of the feature indices
    int start = block_idx * blockDim.x + threadIdx.x;
    int stride = gridDim.y * blockDim.x;  // total threads per sample among blocks

    scalar_t local_sum_pos = static_cast<scalar_t>(0);
    scalar_t local_sum_neg = static_cast<scalar_t>(0);

    // Grid-stride loop over the feature dimension
    for (int f = start; f < feat_size; f += stride) {
        scalar_t a = anchor[base + f];
        scalar_t p = positive[base + f];
        scalar_t n = negative[base + f];
        scalar_t diff_pos = a - p;
        scalar_t diff_neg = a - n;
        local_sum_pos += diff_pos * diff_pos;
        local_sum_neg += diff_neg * diff_neg;
    }

    // Shared memory reduction within the block
    extern __shared__ char shared_mem[];
    scalar_t* sdata_pos = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* sdata_neg = sdata_pos + blockDim.x;
    sdata_pos[threadIdx.x] = local_sum_pos;
    sdata_neg[threadIdx.x] = local_sum_neg;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_pos[threadIdx.x] += sdata_pos[threadIdx.x + s];
            sdata_neg[threadIdx.x] += sdata_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    // One thread per block atomically adds the block's result to the global accumulator
    if (threadIdx.x == 0) {
        atomicAdd(&sum_pos_global[sample], sdata_pos[0]);
        atomicAdd(&sum_neg_global[sample], sdata_neg[0]);
    }
}

// Kernel 2: Final reduction kernel
// Each thread computes the final loss for one sample using the accumulated
// squared distance sums from the partial kernel.

template <typename scalar_t>
__global__ void triplet_margin_loss_final_kernel(
    const scalar_t* __restrict__ sum_pos_global,
    const scalar_t* __restrict__ sum_neg_global,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_size) {

    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batch_size) {
        scalar_t sp = sum_pos_global[sample];
        scalar_t sn = sum_neg_global[sample];
        scalar_t loss = sqrt(sp) - sqrt(sn) + margin;
        output[sample] = loss > 0 ? loss : static_cast<scalar_t>(0);
    }
}

// Host function that launches two kernels to compute the triplet margin loss:
// 1. The partial reduction kernel distributes the workload evenly across the feature dimension
//    using a 2D grid (one dimension for samples, one for blocks per sample).
// 2. The final kernel computes the per-sample loss using the aggregated results.

torch::Tensor evenly_distributed_triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);

    // Number of threads per block for the partial kernel
    const int threads = 256;
    // Determine the number of blocks per sample to cover all feature elements
    int blocksPerSample = (feat_size + threads - 1) / threads;
    // Define a 2D grid: x-dimension for samples, y-dimension for feature-chunk blocks per sample
    dim3 grid(batch_size, blocksPerSample);

    // Allocate global accumulator tensors for each sample's sum of squared differences
    auto sum_pos = torch::zeros({batch_size}, anchor.options());
    auto sum_neg = torch::zeros({batch_size}, anchor.options());

    // Launch the partial reduction kernel
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_partial_kernel", ([&] {
        const int shared_mem_bytes = 2 * threads * sizeof(scalar_t);
        triplet_margin_loss_partial_kernel<scalar_t><<<grid, threads, shared_mem_bytes>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            sum_pos.data_ptr<scalar_t>(),
            sum_neg.data_ptr<scalar_t>(),
            feat_size,
            blocksPerSample);
    }));

    // Allocate output tensor for final loss per sample
    auto loss_tensor = torch::empty({batch_size}, anchor.options());
    
    // Launch the final kernel with one thread per sample using a 1D grid
    int final_threads = 256;
    int final_blocks = (batch_size + final_threads - 1) / final_threads;
    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_final_kernel", ([&] {
        triplet_margin_loss_final_kernel<scalar_t><<<final_blocks, final_threads>>>(
            sum_pos.data_ptr<scalar_t>(),
            sum_neg.data_ptr<scalar_t>(),
            loss_tensor.data_ptr<scalar_t>(),
            margin,
            batch_size);
    }));

    return loss_tensor.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &evenly_distributed_triplet_margin_loss_cuda, "Evenly Distributed Triplet Margin Loss forward (CUDA)");
}
