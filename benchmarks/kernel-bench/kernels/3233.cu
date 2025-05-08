#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

inline int next_power_of_two(int x) {
    return 1 << (32 - __builtin_clz(x - 1));
}

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size,
    const int batch_size) {

    const int tid = threadIdx.x;
    const int group_size = blockDim.x / 8;  // Process 8 elements per block
    const int group_id = tid / group_size;
    const int lane_id = tid % group_size;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* sdata_max = reinterpret_cast<scalar_t*>(smem);
    scalar_t* sdata_sum = sdata_max + 8 * group_size;

    scalar_t max_val[8] = {-INFINITY};
    scalar_t sum_val[8] = {0};

    // Process 8 elements per block with coalesced accesses
    for (int b = 0; b < 8; b++) {
        int batch_idx = blockIdx.x * 8 + b;
        if (batch_idx >= batch_size) break;

        const scalar_t* input_row = input + batch_idx * dim_size;
        scalar_t* output_row = output + batch_idx * dim_size;

        // Find max value
        for (int idx = lane_id; idx < dim_size; idx += group_size) {
            scalar_t val = __ldg(input_row + idx);
            if (val > max_val[b]) max_val[b] = val;
        }

        // Reduce max within warp
        for (int offset = 16; offset > 0; offset /= 2) {
            scalar_t other = __shfl_down_sync(0xffffffff, max_val[b], offset);
            if (other > max_val[b]) max_val[b] = other;
        }

        // Compute sum of exps
        if (lane_id == 0) sdata_max[group_id * 8 + b] = max_val[b];
        __syncthreads();

        scalar_t local_sum = 0;
        for (int idx = lane_id; idx < dim_size; idx += group_size) {
            scalar_t val = exp(__ldg(input_row + idx) - max_val[b]);
            output_row[idx] = val;
            local_sum += val;
        }

        // Reduce sum within warp
        for (int offset = 16; offset > 0; offset /= 2)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

        if (lane_id == 0) sdata_sum[group_id * 8 + b] = local_sum;
        __syncthreads();

        // Final computation
        scalar_t log_sum = log(sdata_sum[group_id * 8 + b]);
        for (int idx = lane_id; idx < dim_size; idx += group_size) {
            output_row[idx] = (__ldg(input_row + idx) - max_val[b]) - log_sum;
        }
    }
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i)
        if (i != dim) permute_dims.push_back(i);
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    const int threads = 1024;  // Max threads per block
    const int blocks = (batch_size + 7) / 8;  // 8 elements per block
    size_t shared_mem = threads * 2 * sizeof(float);  // Max needed for 8 elements

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            batch_size);
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i)
        inverse_permute_dims[permute_dims[i]] = i;
    return output.permute(inverse_permute_dims);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
}
