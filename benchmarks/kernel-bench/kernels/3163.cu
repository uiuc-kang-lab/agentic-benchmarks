#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>
#include <type_traits>


// Kernel using __ldg() for read-only global memory accesses and vectorized load/store on 128-bit boundaries
// For float, we use float4 (4*32 = 128 bits); for double, we use double2 (2*64 = 128 bits).

template <typename scalar_t, int BLOCK_SIZE>
__global__ void optimized_ldg_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

    int row = blockIdx.x;
    const scalar_t* input_row = input + row * dim_size;
    scalar_t* output_row = output + row * dim_size;

    __shared__ scalar_t sdata[BLOCK_SIZE];

    // Determine vectorized parameters
    constexpr int elems_per_vec = (sizeof(scalar_t) == 4) ? 4 : 2;
    int vec_elems = dim_size / elems_per_vec;
    // remainder elements that are not vectorized
    int remainder = dim_size - vec_elems * elems_per_vec;

    // Phase 1: Compute the maximum value using vectorized __ldg() loads
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    if constexpr (std::is_same<scalar_t, float>::value) {
        const float4* vec_ptr = reinterpret_cast<const float4*>(input_row);
        for (int i = threadIdx.x; i < vec_elems; i += BLOCK_SIZE) {
            float4 v = __ldg(&vec_ptr[i]);
            local_max = max(local_max, v.x);
            local_max = max(local_max, v.y);
            local_max = max(local_max, v.z);
            local_max = max(local_max, v.w);
        }
    } else {
        const double2* vec_ptr = reinterpret_cast<const double2*>(input_row);
        for (int i = threadIdx.x; i < vec_elems; i += BLOCK_SIZE) {
            double2 v = __ldg(&vec_ptr[i]);
            local_max = max(local_max, v.x);
            local_max = max(local_max, v.y);
        }
    }

    // Process remaining elements
    for (int i = vec_elems * elems_per_vec + threadIdx.x; i < dim_size; i += BLOCK_SIZE) {
        local_max = max(local_max, __ldg(&input_row[i]));
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Reduction in shared memory to compute the block max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    scalar_t max_val = sdata[0];
    __syncthreads();

    // Phase 2: Compute sum of exponentials exp(x - max_val) using vectorized __ldg() loads
    scalar_t local_sum = 0;
    if constexpr (std::is_same<scalar_t, float>::value) {
        const float4* vec_ptr = reinterpret_cast<const float4*>(input_row);
        for (int i = threadIdx.x; i < vec_elems; i += BLOCK_SIZE) {
            float4 v = __ldg(&vec_ptr[i]);
            local_sum += expf(v.x - max_val);
            local_sum += expf(v.y - max_val);
            local_sum += expf(v.z - max_val);
            local_sum += expf(v.w - max_val);
        }
    } else {
        const double2* vec_ptr = reinterpret_cast<const double2*>(input_row);
        for (int i = threadIdx.x; i < vec_elems; i += BLOCK_SIZE) {
            double2 v = __ldg(&vec_ptr[i]);
            local_sum += exp(v.x - max_val);
            local_sum += exp(v.y - max_val);
        }
    }
    for (int i = vec_elems * elems_per_vec + threadIdx.x; i < dim_size; i += BLOCK_SIZE) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            local_sum += expf(__ldg(&input_row[i]) - max_val);
        } else {
            local_sum += exp(__ldg(&input_row[i]) - max_val);
        }
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduction to compute the total sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    scalar_t sum = sdata[0];
    scalar_t log_sum = (std::is_same<scalar_t, float>::value) ? logf(sum) : log(sum);

    // Phase 3: Write the final LogSoftmax values using vectorized store operations
    if constexpr (std::is_same<scalar_t, float>::value) {
        float4* out_vec_ptr = reinterpret_cast<float4*>(output_row);
        const float4* in_vec_ptr = reinterpret_cast<const float4*>(input_row);
        for (int i = threadIdx.x; i < vec_elems; i += BLOCK_SIZE) {
            float4 v = __ldg(&in_vec_ptr[i]);
            float4 res;
            res.x = (v.x - max_val) - log_sum;
            res.y = (v.y - max_val) - log_sum;
            res.z = (v.z - max_val) - log_sum;
            res.w = (v.w - max_val) - log_sum;
            out_vec_ptr[i] = res;
        }
    } else {
        double2* out_vec_ptr = reinterpret_cast<double2*>(output_row);
        const double2* in_vec_ptr = reinterpret_cast<const double2*>(input_row);
        for (int i = threadIdx.x; i < vec_elems; i += BLOCK_SIZE) {
            double2 v = __ldg(&in_vec_ptr[i]);
            double2 res;
            res.x = (v.x - max_val) - log_sum;
            res.y = (v.y - max_val) - log_sum;
            out_vec_ptr[i] = res;
        }
    }
    for (int i = vec_elems * elems_per_vec + threadIdx.x; i < dim_size; i += BLOCK_SIZE) {
        output_row[i] = (__ldg(&input_row[i]) - max_val) - log_sum;
    }
}


// Host function: Permute tensor to bring the reduction dimension last, launch kernel, and inverse permute

torch::Tensor optimized_128_ldg_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input so that the target dimension is the last dimension
    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            permute_dims.push_back(i);
        }
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    // Choose an optimal block size from {32, 64, 128, 256, 512}
    int optimal_block_size = 256; // default
    if (dim_size <= 32) {
        optimal_block_size = 32;
    } else if (dim_size <= 64) {
        optimal_block_size = 64;
    } else if (dim_size <= 128) {
        optimal_block_size = 128;
    } else if (dim_size <= 256) {
        optimal_block_size = 256;
    } else if (dim_size <= 512) {
        optimal_block_size = 512;
    } else {
        optimal_block_size = 512;
    }

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_128_ldg_logsoftmax_cuda_forward", ([&] {
        if (optimal_block_size == 32) {
            optimized_ldg_logsoftmax_kernel<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 64) {
            optimized_ldg_logsoftmax_kernel<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 128) {
            optimized_ldg_logsoftmax_kernel<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 256) {
            optimized_ldg_logsoftmax_kernel<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (optimal_block_size == 512) {
            optimized_ldg_logsoftmax_kernel<scalar_t, 512><<<blocks, 512>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
    }));

    // Inverse permutation to restore original layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_128_ldg_logsoftmax_cuda_forward, "Optimized 128-bit aligned LogSoftmax forward using __ldg() (CUDA)");
}
