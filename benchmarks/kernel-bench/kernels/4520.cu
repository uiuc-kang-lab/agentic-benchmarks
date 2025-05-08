#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Define maximum channels for constant memory usage
#define MAX_CHANNELS 4096

// Declare constant memory arrays for weight and bias for float and double types
__constant__ float const_weight_float[MAX_CHANNELS];
__constant__ float const_bias_float[MAX_CHANNELS];

__constant__ double const_weight_double[MAX_CHANNELS];
__constant__ double const_bias_double[MAX_CHANNELS];

// Templated device helper functions to get weight and bias from constant memory
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_weight(int idx);

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_bias(int idx);

// Specialization for float
template <>
__device__ __forceinline__ float get_weight<float>(int idx) {
    return const_weight_float[idx];
}

template <>
__device__ __forceinline__ float get_bias<float>(int idx) {
    return const_bias_float[idx];
}

// Specialization for double
template <>
__device__ __forceinline__ double get_weight<double>(int idx) {
    return const_weight_double[idx];
}

template <>
__device__ __forceinline__ double get_bias<double>(int idx) {
    return const_bias_double[idx];
}

// -----------------------------------------------------------------------------
// Compute statistics kernel: computes partial sums in a balanced manner

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const int elements_per_block,
    scalar_t* __restrict__ partial_sums,
    scalar_t* __restrict__ partial_squares) {

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;  // block index for partial sum
    const int n = blockIdx.y;
    const int g = blockIdx.z;
    
    const int group_size = channels_per_group * spatial;
    const int start_idx = bid * elements_per_block;
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    
    __shared__ scalar_t s_sum[256];
    __shared__ scalar_t s_sum_sq[256];
    
    s_sum[tid] = 0;
    s_sum_sq[tid] = 0;
    
    // Each block processes a chunk of the group's elements
    for (int i = tid; i < elements_per_block && (start_idx + i) < group_size; i += blockDim.x) {
        const int global_idx = group_offset + start_idx + i;
        scalar_t val = x[global_idx];
        s_sum[tid] += val;
        s_sum_sq[tid] += val * val;
    }
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Store the partial result for this block
        const int partial_idx = n * (num_groups * gridDim.x) + g * gridDim.x + bid;
        partial_sums[partial_idx] = s_sum[0];
        partial_squares[partial_idx] = s_sum_sq[0];
    }
}

// -----------------------------------------------------------------------------
// Finalize statistics kernel: aggregates partial results to compute mean and variance

template <typename scalar_t>
__global__ void finalize_stats_kernel(
    const scalar_t* __restrict__ partial_sums,
    const scalar_t* __restrict__ partial_squares,
    const int num_blocks,
    const int group_size,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

    const int n = blockIdx.y;
    const int g = threadIdx.x;
    
    scalar_t sum = 0;
    scalar_t sum_sq = 0;
    
    // Aggregate partial sums from all blocks for group (n, g)
    for (int i = 0; i < num_blocks; i++) {
        const int idx = n * (gridDim.x * num_blocks) + g * num_blocks + i;
        sum += partial_sums[idx];
        sum_sq += partial_squares[idx];
    }
    
    const scalar_t group_mean = sum / group_size;
    const scalar_t group_var = sum_sq / group_size - group_mean * group_mean;
    const int out_idx = n * gridDim.x + g;
    mean[out_idx] = group_mean;
    var[out_idx] = group_var;
}

// -----------------------------------------------------------------------------
// Group normalization forward kernel using constant memory for weight and bias

template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * spatial;
    if (idx >= total) return;
    
    // Decode the flattened index into (n, c, j) coordinates
    const int j = idx % spatial;
    const int temp = idx / spatial;
    const int c = temp % C;
    const int n = temp / C;
    
    const int g = c / channels_per_group;
    const int stats_idx = n * num_groups + g;
    
    scalar_t m = mean[stats_idx];
    scalar_t v = var[stats_idx];
    scalar_t inv_std = rsqrt(v + eps);
    scalar_t x_val = x[idx];
    
    // Access weight and bias from constant memory
    scalar_t w = get_weight<scalar_t>(c);
    scalar_t b = get_bias<scalar_t>(c);
    
    y[idx] = (x_val - m) * inv_std * w + b;
}

// -----------------------------------------------------------------------------
// Host function: Group normalization forward pass

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
    
    // x is expected to have shape (N, C, ...)
    const int N = x.size(0);
    const int C = x.size(1);
    int spatial = 1;
    for (int i = 2; i < x.dim(); i++) {
        spatial *= x.size(i);
    }
    
    const int channels_per_group = C / num_groups;
    const int group_size = channels_per_group * spatial;
    
    // Define execution configuration for stats kernels
    const int block_size = 256;
    const int elements_per_block = 1024;
    const int num_blocks = (group_size + elements_per_block - 1) / elements_per_block;
    
    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);
    auto partial_sums = torch::empty({N, num_groups, num_blocks}, options);
    auto partial_squares = torch::empty({N, num_groups, num_blocks}, options);
    
    // Grid configuration for compute_stats_kernel: (num_blocks, N, num_groups)
    dim3 stats_blocks(num_blocks, N, num_groups);
    
    // Grid configuration for group_norm_forward_kernel
    int total_elements = N * C * spatial;
    dim3 norm_blocks((total_elements + block_size - 1) / block_size);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        // Copy weight and bias to constant memory based on the data type
        if (std::is_same<scalar_t, float>::value) {
            cudaMemcpyToSymbol(const_weight_float, weight.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(const_bias_float, bias.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyToSymbol(const_weight_double, weight.data_ptr<double>(), C * sizeof(double), 0, cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(const_bias_double, bias.data_ptr<double>(), C * sizeof(double), 0, cudaMemcpyDeviceToDevice);
        }
        
        // Launch kernel to compute partial sums for mean/variance
        compute_stats_kernel<scalar_t><<<stats_blocks, block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            N,
            C,
            spatial,
            channels_per_group,
            num_groups,
            elements_per_block,
            partial_sums.data_ptr<scalar_t>(),
            partial_squares.data_ptr<scalar_t>());
        
        // Launch kernel to aggregate partial sums and finalize mean/variance
        finalize_stats_kernel<scalar_t><<<dim3(1, N), num_groups, 0, stream>>>(
            partial_sums.data_ptr<scalar_t>(),
            partial_squares.data_ptr<scalar_t>(),
            num_blocks,
            group_size,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());
        
        // Launch normalization kernel using constant memory for weight and bias
        group_norm_forward_kernel<scalar_t><<<norm_blocks, block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            N,
            C,
            spatial,
            channels_per_group,
            num_groups,
            static_cast<scalar_t>(eps),
            y.data_ptr<scalar_t>());
    }));
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA)");
}
