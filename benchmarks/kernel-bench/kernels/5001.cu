#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Stage 1: Reduction kernel, computing partial sum of squares
template <typename scalar_t>
__global__ void l2_normalize_kernel_stage1(
    const scalar_t* __restrict__ input,
    scalar_t* partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int elements_per_block) {

    // Determine which vector and segment this block works on
    int segments_per_vector = (C + elements_per_block - 1) / elements_per_block;
    const int vector_idx = blockIdx.x / segments_per_vector;
    const int block_segment = blockIdx.x % segments_per_vector;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int seg_start = block_segment * elements_per_block;
    int seg_end = seg_start + elements_per_block;
    if (seg_end > C) seg_end = C;
    const int num_elements = seg_end - seg_start;

    scalar_t thread_sum = 0;

    // If the inner stride is 1, we can use aligned, vectorized loads
    if (stride_C == 1) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            // Process elements in groups of 4 (128-bit load)
            int vec_elems = (num_elements / 4) * 4;
            const float4* input_vec = reinterpret_cast<const float4*>(input + base_offset + seg_start);
            for (int i = threadIdx.x * 4; i < vec_elems; i += blockDim.x * 4) {
                float4 data = __ldg(input_vec + i/4);
                thread_sum += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
            }
            // Process remainder
            for (int i = vec_elems + threadIdx.x; i < num_elements; i += blockDim.x) {
                float val = __ldg(input + base_offset + seg_start + i);
                thread_sum += val * val;
            }
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            // Process elements in groups of 2 (128-bit load)
            int vec_elems = (num_elements / 2) * 2;
            const double2* input_vec = reinterpret_cast<const double2*>(input + base_offset + seg_start);
            for (int i = threadIdx.x * 2; i < vec_elems; i += blockDim.x * 2) {
                double2 data = __ldg(input_vec + i/2);
                thread_sum += data.x * data.x + data.y * data.y;
            }
            // Process remainder
            for (int i = vec_elems + threadIdx.x; i < num_elements; i += blockDim.x) {
                double val = __ldg(input + base_offset + seg_start + i);
                thread_sum += val * val;
            }
        } else {
            // Fallback for other types: use scalar loads with __ldg
            for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
                const scalar_t val = __ldg(input + base_offset + seg_start + i);
                thread_sum += val * val;
            }
        }
    } else {
        // Non-unit stride: fallback to scalar loading
        for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
            int idx = seg_start + i;
            const scalar_t val = __ldg(input + base_offset + idx * stride_C);
            thread_sum += val * val;
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Use shared memory for block-level reduction (one value per warp)
    __shared__ scalar_t shared_mem[32];
    if ((threadIdx.x % 32) == 0) {
        shared_mem[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();

    // Final reduction among warp sums
    if (threadIdx.x < 32) {
        thread_sum = shared_mem[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(&partial_sums[vector_idx], thread_sum);
        }
    }
}

// Stage 2: Normalization kernel, scaling the input by the inverse norm
template <typename scalar_t>
__global__ void l2_normalize_kernel_stage2(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const scalar_t* partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int elements_per_block) {

    int segments_per_vector = (C + elements_per_block - 1) / elements_per_block;
    const int vector_idx = blockIdx.x / segments_per_vector;
    const int block_segment = blockIdx.x % segments_per_vector;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int seg_start = block_segment * elements_per_block;
    int seg_end = seg_start + elements_per_block;
    if (seg_end > C) seg_end = C;
    const int num_elements = seg_end - seg_start;

    // Compute inverse norm
    scalar_t norm = partial_sums[vector_idx];
    scalar_t inv_norm = 1.0 / (sqrt(norm) + 1e-12);

    if (stride_C == 1) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            int vec_elems = (num_elements / 4) * 4;
            const float4* input_vec = reinterpret_cast<const float4*>(input + base_offset + seg_start);
            float4* output_vec = reinterpret_cast<float4*>(output + base_offset + seg_start);
            for (int i = threadIdx.x * 4; i < vec_elems; i += blockDim.x * 4) {
                float4 data = __ldg(input_vec + i/4);
                float4 res;
                res.x = data.x * inv_norm;
                res.y = data.y * inv_norm;
                res.z = data.z * inv_norm;
                res.w = data.w * inv_norm;
                output_vec[i/4] = res;
            }
            for (int i = vec_elems + threadIdx.x; i < num_elements; i += blockDim.x) {
                float val = __ldg(input + base_offset + seg_start + i);
                output[base_offset + seg_start + i] = val * inv_norm;
            }
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            int vec_elems = (num_elements / 2) * 2;
            const double2* input_vec = reinterpret_cast<const double2*>(input + base_offset + seg_start);
            double2* output_vec = reinterpret_cast<double2*>(output + base_offset + seg_start);
            for (int i = threadIdx.x * 2; i < vec_elems; i += blockDim.x * 2) {
                double2 data = __ldg(input_vec + i/2);
                double2 res;
                res.x = data.x * inv_norm;
                res.y = data.y * inv_norm;
                output_vec[i/2] = res;
            }
            for (int i = vec_elems + threadIdx.x; i < num_elements; i += blockDim.x) {
                double val = __ldg(input + base_offset + seg_start + i);
                output[base_offset + seg_start + i] = val * inv_norm;
            }
        } else {
            for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
                const scalar_t val = __ldg(input + base_offset + seg_start + i);
                output[base_offset + seg_start + i] = val * inv_norm;
            }
        }
    } else {
        for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
            int idx = seg_start + i;
            const scalar_t val = __ldg(input + base_offset + idx * stride_C);
            output[base_offset + idx * stride_C] = val * inv_norm;
        }
    }
}

// C++ interface: launched from PyTorch
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto partial_sums = torch::zeros({total_vectors}, input.options());

    const int threads = 256;
    const int elements_per_block = 1024;
    const int segments_per_vector = (C + elements_per_block - 1) / elements_per_block;
    const int total_blocks = segments_per_vector * total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_ldg_aligned", [&] {
        l2_normalize_kernel_stage1<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            elements_per_block
        );

        l2_normalize_kernel_stage2<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            elements_per_block
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with __ldg and 128-bit aligned memory accesses");
}
