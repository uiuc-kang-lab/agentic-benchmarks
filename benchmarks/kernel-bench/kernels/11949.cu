#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Define maximum number of elements allowed in constant memory
// Ensure that batch_size * feat_size <= MAX_CONSTANT_MEMORY_SIZE
#define MAX_CONSTANT_MEMORY_SIZE 65536

// Declare constant memory for float data
__constant__ float c_anchor_f[MAX_CONSTANT_MEMORY_SIZE];
__constant__ float c_positive_f[MAX_CONSTANT_MEMORY_SIZE];
__constant__ float c_negative_f[MAX_CONSTANT_MEMORY_SIZE];

// Declare constant memory for double data
__constant__ double c_anchor_d[MAX_CONSTANT_MEMORY_SIZE];
__constant__ double c_positive_d[MAX_CONSTANT_MEMORY_SIZE];
__constant__ double c_negative_d[MAX_CONSTANT_MEMORY_SIZE];

// Kernel for float using constant memory
__global__ void constant_triplet_margin_loss_kernel_float(
    float* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int base_idx = batch_idx * feat_size;
    float local_dist_pos = 0.0f;
    float local_dist_neg = 0.0f;

    // Each thread processes a subset of the features
    for (int i = tid; i < feat_size; i += blockDim.x) {
        int idx = base_idx + i;
        float a = c_anchor_f[idx];
        float p = c_positive_f[idx];
        float n = c_negative_f[idx];
        float diff_pos = a - p;
        float diff_neg = a - n;
        local_dist_pos += diff_pos * diff_pos;
        local_dist_neg += diff_neg * diff_neg;
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
        local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
    }

    __shared__ float shared_mem_pos[32];
    __shared__ float shared_mem_neg[32];

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        shared_mem_pos[warp_id] = local_dist_pos;
        shared_mem_neg[warp_id] = local_dist_neg;
    }
    __syncthreads();

    float sum_pos = 0.0f;
    float sum_neg = 0.0f;

    if (tid < (blockDim.x / 32)) {
        sum_pos = shared_mem_pos[lane];
        sum_neg = shared_mem_neg[lane];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_pos += __shfl_down_sync(0xffffffff, sum_pos, offset);
        sum_neg += __shfl_down_sync(0xffffffff, sum_neg, offset);
    }

    if (lane == 0 && warp_id == 0) {
        float loss = sqrtf(sum_pos) - sqrtf(sum_neg) + margin;
        output[batch_idx] = loss < 0.0f ? 0.0f : loss;
    }
}

// Kernel for double using constant memory
__global__ void constant_triplet_margin_loss_kernel_double(
    double* __restrict__ output,
    const double margin,
    const int batch_size,
    const int feat_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int base_idx = batch_idx * feat_size;
    double local_dist_pos = 0.0;
    double local_dist_neg = 0.0;

    // Each thread accumulates over a portion of the feature vector
    for (int i = tid; i < feat_size; i += blockDim.x) {
        int idx = base_idx + i;
        double a = c_anchor_d[idx];
        double p = c_positive_d[idx];
        double n = c_negative_d[idx];
        double diff_pos = a - p;
        double diff_neg = a - n;
        local_dist_pos += diff_pos * diff_pos;
        local_dist_neg += diff_neg * diff_neg;
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_dist_pos += __shfl_down_sync(0xffffffff, local_dist_pos, offset);
        local_dist_neg += __shfl_down_sync(0xffffffff, local_dist_neg, offset);
    }

    __shared__ double shared_mem_pos[32];
    __shared__ double shared_mem_neg[32];

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        shared_mem_pos[warp_id] = local_dist_pos;
        shared_mem_neg[warp_id] = local_dist_neg;
    }
    __syncthreads();

    double sum_pos = 0.0;
    double sum_neg = 0.0;

    if (tid < (blockDim.x / 32)) {
        sum_pos = shared_mem_pos[lane];
        sum_neg = shared_mem_neg[lane];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_pos += __shfl_down_sync(0xffffffff, sum_pos, offset);
        sum_neg += __shfl_down_sync(0xffffffff, sum_neg, offset);
    }

    if (lane == 0 && warp_id == 0) {
        double loss = sqrt(sum_pos) - sqrt(sum_neg) + margin;
        output[batch_idx] = loss < 0.0 ? 0.0 : loss;
    }
}

// Host interface
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
    TORCH_CHECK(batch_size * feat_size <= MAX_CONSTANT_MEMORY_SIZE, "Data exceeds constant memory limits");

    auto output = torch::zeros({batch_size}, anchor.options());

    const int threads = 256;
    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_cuda_const", ([&] {
        if (std::is_same<scalar_t, float>::value) {
            // Copy float data to constant memory
            cudaMemcpyToSymbol(c_anchor_f, anchor.data_ptr<scalar_t>(), batch_size * feat_size * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(c_positive_f, positive.data_ptr<scalar_t>(), batch_size * feat_size * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(c_negative_f, negative.data_ptr<scalar_t>(), batch_size * feat_size * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
            
            constant_triplet_margin_loss_kernel_float<<<blocks, threads>>>(
                output.data_ptr<scalar_t>(), margin, batch_size, feat_size);
        } else {
            // Copy double data to constant memory
            cudaMemcpyToSymbol(c_anchor_d, anchor.data_ptr<scalar_t>(), batch_size * feat_size * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(c_positive_d, positive.data_ptr<scalar_t>(), batch_size * feat_size * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(c_negative_d, negative.data_ptr<scalar_t>(), batch_size * feat_size * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
            
            constant_triplet_margin_loss_kernel_double<<<blocks, threads>>>(
                output.data_ptr<scalar_t>(), margin, batch_size, feat_size);
        }
    }));

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA) using constant memory");
}
