#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

// Warp-level reduction using shuffle intrinsics
template <typename scalar_t>
__device__ inline void warp_reduce_sum(scalar_t &val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

// CUDA kernel with vectorized loads and __ldg() for read-only global memory accesses
template <typename scalar_t>
__global__ void triplet_margin_loss_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warpSize;  // one warp processes one sample
    const int lane = tid % warpSize;

    if (warp_id >= batch_size) return;

    const int sample_base = warp_id * feat_size;
    scalar_t accum_pos = 0;
    scalar_t accum_neg = 0;

    // Determine the number of elements we can load in a 128-bit (16-byte) chunk
    int vec_elems = 1;  // fallback to scalar loads
    if constexpr (std::is_same<scalar_t, float>::value) {
        vec_elems = 4; // 4 floats * 4 bytes = 16 bytes
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        vec_elems = 2; // 2 doubles * 8 bytes = 16 bytes
    }
    int nvec = feat_size / vec_elems;

    // Process vectorized chunks using __ldg for read-only accesses
    for (int i = lane; i < nvec; i += warpSize) {
        int idx = sample_base + i * vec_elems;
        if constexpr (std::is_same<scalar_t, float>::value) {
            const float4 a_vec = __ldg(reinterpret_cast<const float4*>(anchor + idx));
            const float4 p_vec = __ldg(reinterpret_cast<const float4*>(positive + idx));
            const float4 n_vec = __ldg(reinterpret_cast<const float4*>(negative + idx));
            
            float diff_a0 = a_vec.x - p_vec.x;
            float diff_a1 = a_vec.y - p_vec.y;
            float diff_a2 = a_vec.z - p_vec.z;
            float diff_a3 = a_vec.w - p_vec.w;
            accum_pos += diff_a0 * diff_a0 + diff_a1 * diff_a1 + diff_a2 * diff_a2 + diff_a3 * diff_a3;
            
            float diff_n0 = a_vec.x - n_vec.x;
            float diff_n1 = a_vec.y - n_vec.y;
            float diff_n2 = a_vec.z - n_vec.z;
            float diff_n3 = a_vec.w - n_vec.w;
            accum_neg += diff_n0 * diff_n0 + diff_n1 * diff_n1 + diff_n2 * diff_n2 + diff_n3 * diff_n3;
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            const double2 a_vec = __ldg(reinterpret_cast<const double2*>(anchor + idx));
            const double2 p_vec = __ldg(reinterpret_cast<const double2*>(positive + idx));
            const double2 n_vec = __ldg(reinterpret_cast<const double2*>(negative + idx));
            
            double diff_a0 = a_vec.x - p_vec.x;
            double diff_a1 = a_vec.y - p_vec.y;
            accum_pos += diff_a0 * diff_a0 + diff_a1 * diff_a1;
            
            double diff_n0 = a_vec.x - n_vec.x;
            double diff_n1 = a_vec.y - n_vec.y;
            accum_neg += diff_n0 * diff_n0 + diff_n1 * diff_n1;
        } else {
            // Fallback for other types
            for (int j = 0; j < vec_elems; j++) {
                scalar_t a = __ldg(anchor + idx + j);
                scalar_t p = __ldg(positive + idx + j);
                scalar_t n = __ldg(negative + idx + j);
                scalar_t diff_a = a - p;
                accum_pos += diff_a * diff_a;
                scalar_t diff_n = a - n;
                accum_neg += diff_n * diff_n;
            }
        }
    }

    // Process any remaining elements that don't fit in a full vectorized load
    int rem_start = nvec * vec_elems;
    for (int i = sample_base + rem_start + lane; i < sample_base + feat_size; i += warpSize) {
        scalar_t a = __ldg(anchor + i);
        scalar_t p = __ldg(positive + i);
        scalar_t n = __ldg(negative + i);
        scalar_t diff_a = a - p;
        accum_pos += diff_a * diff_a;
        scalar_t diff_n = a - n;
        accum_neg += diff_n * diff_n;
    }

    // Reduce within the warp
    warp_reduce_sum(accum_pos);
    warp_reduce_sum(accum_neg);

    if (lane == 0) {
        scalar_t sqrt_pos;
        scalar_t sqrt_neg;
        if constexpr (std::is_same<scalar_t, float>::value) {
            sqrt_pos = sqrtf(accum_pos);
            sqrt_neg = sqrtf(accum_neg);
        } else {
            sqrt_pos = sqrt(accum_pos);
            sqrt_neg = sqrt(accum_neg);
        }
        scalar_t loss = fmaxf(0.0, sqrt_pos - sqrt_neg + static_cast<scalar_t>(margin));
        output[warp_id] = loss;
    }
}

// Host function to launch the CUDA kernel
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

    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / warpSize;
    const int blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel", ([&] {
        triplet_margin_loss_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
