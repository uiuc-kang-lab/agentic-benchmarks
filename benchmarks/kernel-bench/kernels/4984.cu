#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel performs L2 normalization ensuring memory coalescing by aligning global memory accesses.
// When the data is contiguous (stride_C == 1), it uses vectorized loads/stores (float4 for float, double2 for double)
// so that threads in a warp read/write consecutive memory locations. For non-contiguous cases, a fallback loop is used.

template <typename scalar_t>
__global__ void l2norm_aligned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    // Each block processes one vector
    int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    int base = vector_idx * outer_stride;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    scalar_t sum = 0;

    // Coalesced reads: if stride_C == 1, use vectorized loads
    if (stride_C == 1) {
        // factor: 4 for float, 2 for double
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        int aligned_C = (C / factor) * factor;
        
        if constexpr (sizeof(scalar_t) == 4) {
            // Use float4 for vectorized loads
            const float4* in_vec = reinterpret_cast<const float4*>(input + base);
            int num_vec = aligned_C / 4;
            for (int i = tid; i < num_vec; i += block_size) {
                float4 v = in_vec[i];
                sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            // Use double2 for vectorized loads
            const double2* in_vec = reinterpret_cast<const double2*>(input + base);
            int num_vec = aligned_C / 2;
            for (int i = tid; i < num_vec; i += block_size) {
                double2 v = in_vec[i];
                sum += v.x * v.x + v.y * v.y;
            }
        }
        // Process any remaining elements
        for (int i = aligned_C + tid; i < C; i += block_size) {
            scalar_t val = input[base + i];
            sum += val * val;
        }
    } else {
        // Fallback for non-contiguous data
        for (int i = tid; i < C; i += block_size) {
            scalar_t val = input[base + i * stride_C];
            sum += val * val;
        }
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ scalar_t shared[256];
    int lane = tid % warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) shared[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        sum = (lane < (block_size + warpSize - 1) / warpSize) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
    }

    // Thread 0 computes normalization factor and writes normalized values
    if (tid == 0) {
        scalar_t norm = sqrt(sum) + (scalar_t)1e-12;
        scalar_t inv_norm = (scalar_t)1.0 / norm;
        
        // Write back normalized vector ensuring coalesced writes
        if (stride_C == 1) {
            const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
            int aligned_C = (C / factor) * factor;
            if constexpr (sizeof(scalar_t) == 4) {
                float4* out_vec = reinterpret_cast<float4*>(output + base);
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                int num_vec = aligned_C / 4;
                for (int i = 0; i < num_vec; i++) {
                    float4 v = in_vec[i];
                    v.x *= inv_norm;
                    v.y *= inv_norm;
                    v.z *= inv_norm;
                    v.w *= inv_norm;
                    out_vec[i] = v;
                }
            } else {
                double2* out_vec = reinterpret_cast<double2*>(output + base);
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                int num_vec = aligned_C / 2;
                for (int i = 0; i < num_vec; i++) {
                    double2 v = in_vec[i];
                    v.x *= inv_norm;
                    v.y *= inv_norm;
                    out_vec[i] = v;
                }
            }
            for (int i = aligned_C; i < C; i++) {
                output[base + i] = input[base + i] * inv_norm;
            }
        } else {
            // Fallback for non-contiguous writes
            for (int i = 0; i < C; i++) {
                output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
            }
        }
    }
}

// Host forward function
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    int C = input.size(1);
    int total_vectors = input.numel() / C;
    int stride_C = input.stride(1);
    int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const dim3 blocks(total_vectors);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_aligned_kernel", ([&] {
        l2norm_aligned_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with globally aligned memory accesses for coalescing");
}
