#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

static const int BLOCK_SIZE = 256;
static const int N_STREAMS = 4;
static const int MIN_ELEMENTS_PER_STREAM = 10000;

template <typename scalar_t>
__global__ void hybrid_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_result,
    const int64_t start_idx,
    const int64_t seg_size
) {
    __shared__ double sdata[BLOCK_SIZE];
    const int tid = threadIdx.x;
    int idx = start_idx + blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    const int end_idx = start_idx + seg_size;
    
    using vec2 = typename std::aligned_storage<sizeof(scalar_t[2]), alignof(scalar_t[2])>::type;
    const vec2* preds_vec = reinterpret_cast<const vec2*>(preds + idx);
    const vec2* tgts_vec = reinterpret_cast<const vec2*>(tgts + idx);
    
    double sum_val = 0.0;
    
    while (idx < end_idx) {
        if (idx + 1 < end_idx && (idx & 1) == 0) {
            scalar_t p[2], t[2];
            *reinterpret_cast<vec2*>(p) = *preds_vec;
            *reinterpret_cast<vec2*>(t) = *tgts_vec;
            
            double diff0 = static_cast<double>(p[0]) - static_cast<double>(t[0]);
            double diff1 = static_cast<double>(p[1]) - static_cast<double>(t[1]);
            sum_val += diff0 * diff0 + diff1 * diff1;
            
            idx += 2;
            preds_vec++;
            tgts_vec++;
        } else {
            double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
            sum_val += diff * diff;
            idx++;
        }
        idx += grid_stride - 1;
    }
    
    sdata[tid] = sum_val;
    __syncthreads();
    
    #pragma unroll
    for (int offset = BLOCK_SIZE/2; offset > 32; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile double* smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] += smem[tid + 32];
        if (BLOCK_SIZE >= 32) smem[tid] += smem[tid + 16];
        if (BLOCK_SIZE >= 16) smem[tid] += smem[tid + 8];
        if (BLOCK_SIZE >= 8) smem[tid] += smem[tid + 4];
        if (BLOCK_SIZE >= 4) smem[tid] += smem[tid + 2];
        if (BLOCK_SIZE >= 2) smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        atomicAdd(partial_result, sdata[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    
    int streams_to_use = std::min(N_STREAMS, 
                                 (int)(num_elements / MIN_ELEMENTS_PER_STREAM) + 1);
    streams_to_use = std::max(1, streams_to_use);
    
    auto partial_results = torch::zeros({streams_to_use}, 
                                      predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();

    std::vector<cudaStream_t> streams(streams_to_use);
    for (int i = 0; i < streams_to_use; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int64_t chunk_size = (num_elements + streams_to_use - 1) / streams_to_use;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "hybrid_mse_cuda", ([&] {
        const scalar_t* preds_ptr = predictions.data_ptr<scalar_t>();
        const scalar_t* tgts_ptr = targets.data_ptr<scalar_t>();
        
        for (int i = 0; i < streams_to_use; i++) {
            int64_t start_idx = i * chunk_size;
            int64_t seg_size = std::min(chunk_size, num_elements - start_idx);
            if (seg_size <= 0) continue;
            
            int grid_size = std::min(1024, (int)((seg_size + BLOCK_SIZE - 1) / BLOCK_SIZE));
            
            hybrid_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
                preds_ptr, tgts_ptr, d_partial + i, start_idx, seg_size
            );
        }
    }));

    auto final_sum = partial_results.sum();
    double mse = final_sum.item<double>() / static_cast<double>(num_elements);
    
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }

    return torch::full({1}, mse, predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid MSE forward (CUDA)");
}