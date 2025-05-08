#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename scalar_t, int BLOCK_SIZE>
__global__ void balanced_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    
    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // Compute chunk size per thread for balanced workload
    const int vec_size = 4;  // Using float4 for coalesced memory access
    const int elements_per_thread = (dim_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    const int vec_elements = elements_per_thread / vec_size;
    
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    // Phase 1: Find maximum using vectorized loads
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    
    const float4* input_vec = reinterpret_cast<const float4*>(input_row);
    #pragma unroll 4
    for (int i = 0; i < vec_elements; i++) {
        const int idx = tid * vec_elements + i;
        if (idx * vec_size < dim_size) {
            float4 values = input_vec[idx];
            thread_max = max(thread_max, max(max(values.x, values.y), max(values.z, values.w)));
        }
    }
    
    // Handle remaining elements
    const int rem_start = vec_elements * vec_size;
    for (int i = rem_start + tid; i < dim_size; i += BLOCK_SIZE) {
        thread_max = max(thread_max, input_row[i]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, warp.shfl_down(thread_max, offset));
    }

    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_data[warp_id] = thread_max;
    }
    block.sync();

    // Final reduction across warps
    if (tid < (BLOCK_SIZE / warpSize)) {
        scalar_t warp_max = shared_data[tid];
        #pragma unroll
        for (int i = tid + 1; i < BLOCK_SIZE / warpSize; i++) {
            warp_max = max(warp_max, shared_data[i]);
        }
        shared_data[tid] = warp_max;
    }
    block.sync();

    const scalar_t max_val = shared_data[0];

    // Phase 2: Compute sum of exponentials using vectorized operations
    scalar_t thread_sum = 0;
    
    #pragma unroll 4
    for (int i = 0; i < vec_elements; i++) {
        const int idx = tid * vec_elements + i;
        if (idx * vec_size < dim_size) {
            float4 values = input_vec[idx];
            thread_sum += exp(values.x - max_val);
            thread_sum += exp(values.y - max_val);
            thread_sum += exp(values.z - max_val);
            thread_sum += exp(values.w - max_val);
        }
    }

    for (int i = rem_start + tid; i < dim_size; i += BLOCK_SIZE) {
        thread_sum += exp(input_row[i] - max_val);
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += warp.shfl_down(thread_sum, offset);
    }

    if (lane_id == 0) {
        shared_data[warp_id] = thread_sum;
    }
    block.sync();

    // Final reduction and compute log sum
    if (tid == 0) {
        scalar_t total_sum = 0;
        for (int i = 0; i < BLOCK_SIZE / warpSize; i++) {
            total_sum += shared_data[i];
        }
        shared_data[0] = log(total_sum);
    }
    block.sync();

    const scalar_t log_sum = shared_data[0];

    // Phase 3: Compute final values using vectorized operations
    float4* output_vec = reinterpret_cast<float4*>(output_row);
    
    #pragma unroll 4
    for (int i = 0; i < vec_elements; i++) {
        const int idx = tid * vec_elements + i;
        if (idx * vec_size < dim_size) {
            float4 values = input_vec[idx];
            float4 result;
            result.x = (values.x - max_val) - log_sum;
            result.y = (values.y - max_val) - log_sum;
            result.z = (values.z - max_val) - log_sum;
            result.w = (values.w - max_val) - log_sum;
            output_vec[idx] = result;
        }
    }

    for (int i = rem_start + tid; i < dim_size; i += BLOCK_SIZE) {
        output_row[i] = (input_row[i] - max_val) - log_sum;
    }
}

torch::Tensor balanced_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

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

    // Choose block size based on dimension size for better occupancy
    int optimal_block_size = 256;
    if (dim_size <= 128) {
        optimal_block_size = 128;
    } else if (dim_size <= 256) {
        optimal_block_size = 256;
    } else {
        optimal_block_size = 512;
    }

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "balanced_logsoftmax_cuda_forward", ([&] {
        switch(optimal_block_size) {
            case 128:
                balanced_logsoftmax_kernel<scalar_t, 128><<<blocks, 128>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size);
                break;
            case 256:
                balanced_logsoftmax_kernel<scalar_t, 256><<<blocks, 256>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size);
                break;
            default:
                balanced_logsoftmax_kernel<scalar_t, 512><<<blocks, 512>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size);
        }
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_logsoftmax_cuda_forward, "Balanced LogSoftmax forward (CUDA)");
}