#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename scalar_t, int BLOCK_SIZE = 256, int WARP_SIZE = 32>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    if (tid >= outer_size * inner_size) return;

    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int64_t offset = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t sum = 0;
    
    if (inner_size == 1) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            if ((dim_size % 4 == 0) && ((((uintptr_t)(input + offset)) & 0xF) == 0)) {
                const float4* vec_input = reinterpret_cast<const float4*>(input + offset);
                #pragma unroll
                for (int i = 0; i < dim_size / 4; i++) {
                    float4 val = __ldg(vec_input + i);
                    sum += val.x + val.y + val.z + val.w;
                }
            } else {
                #pragma unroll 4
                for (int i = 0; i < dim_size; i++) {
                    sum += __ldg(input + offset + i);
                }
            }
        } else {
            #pragma unroll 4
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(input + offset + i);
            }
        }
    } else {
        #pragma unroll 4
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(input + offset + i * inner_size);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        shared_data[warp_id] = sum;
    }

    if (blockDim.x > WARP_SIZE) {
        __syncthreads();
    }

    if (warp_id == 0 && lane_id < (blockDim.x / WARP_SIZE)) {
        sum = shared_data[lane_id];
        
        #pragma unroll
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            output[blockIdx.x] = sum / static_cast<scalar_t>(dim_size);
        }
    }
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}