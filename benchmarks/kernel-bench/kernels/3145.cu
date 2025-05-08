#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Store the dimension size in constant memory for fast read-only access
__constant__ int d_dim_size;

// CUDA kernel using constant memory for dim size
template <typename scalar_t>
__global__ void log_softmax_forward_kernel_const(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output) {

    // Each block handles one batch element (row)
    int batch_idx = blockIdx.x;

    // Compute pointer offsets using the constant dimension size
    const scalar_t* input_row = input + batch_idx * d_dim_size;
    scalar_t* output_row = output + batch_idx * d_dim_size;

    // Allocate shared memory for reductions
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Compute the maximum value in the row
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < d_dim_size; idx += blockDim.x) {
        scalar_t val = input_row[idx];
        max_val = max(max_val, val);
    }

    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // Reduce maximum value using tree reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < d_dim_size) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Compute the sum of exp(input - max_val) in a numerically stable manner
    scalar_t sum = 0;
    for (int idx = threadIdx.x; idx < d_dim_size; idx += blockDim.x) {
        scalar_t exp_val = exp(input_row[idx] - max_val);
        // Temporarily store exp value in output for reuse
        output_row[idx] = exp_val;
        sum += exp_val;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce sum in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < d_dim_size) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    scalar_t log_sum = log(sum);

    // Final computation of LogSoftmax
    for (int idx = threadIdx.x; idx < d_dim_size; idx += blockDim.x) {
        output_row[idx] = (input_row[idx] - max_val) - log_sum;
    }
}

// Host function that launches the CUDA kernel
torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input to bring the reduce dimension to the last dimension
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

    // Copy the dimension size to constant memory for fast read-only access
    cudaMemcpyToSymbol(d_dim_size, &dim_size, sizeof(int));

    auto output = torch::empty_like(input);

    // Determine the number of threads as the next power of 2 (capped to 1024)
    int threads = 1;
    while (threads < dim_size) threads <<= 1;
    if (threads > 1024) threads = 1024;
    size_t shared_mem_size = threads * sizeof(float);
    int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        size_t smem_size = threads * sizeof(scalar_t);
        log_softmax_forward_kernel_const<scalar_t><<<blocks, threads, smem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
    }));

    // Inverse permute to restore the original tensor shape
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA) with constant memory");
}
