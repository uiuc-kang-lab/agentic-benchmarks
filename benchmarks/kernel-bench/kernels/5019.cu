#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Stage 1: Compute the squared sum for each vector in a coalesced manner
// Assumes input is a contiguous 2D tensor with shape [total_vectors, C]

template <typename scalar_t>
__global__ void l2norm_coalesced_stage1(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ partial_sums,
    const int C) {

    int vecIdx = blockIdx.x;  // one block per vector
    if (vecIdx >= gridDim.x) return;

    // Get pointer to the beginning of the vector
    const scalar_t* vec = input + vecIdx * C;
    scalar_t sum = 0.0;

    // Each thread processes a contiguous stripe of the vector
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        scalar_t x = vec[i];
        sum += x * x;
    }

    // Warp-level reduction using shuffle instructions for coalesced communication
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to reduce across warps in the block
    __shared__ scalar_t shared[32];  // Enough to hold one value per warp
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[wid] = sum;
    }
    __syncthreads();

    // First warp reduces the partial sums from each warp
    if (threadIdx.x < warpSize) {
        sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[threadIdx.x] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            partial_sums[vecIdx] = sum;
        }
    }
}

// Stage 2: Normalize each vector using the computed norm
// Assumes output tensor is contiguous and uses the same layout as input

template <typename scalar_t>
__global__ void l2norm_coalesced_stage2(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ partial_sums,
    const int C) {

    int vecIdx = blockIdx.x;  // one block per vector
    if (vecIdx >= gridDim.x) return;

    scalar_t norm = partial_sums[vecIdx];
    scalar_t invNorm = 1.0 / (sqrt(norm) + 1e-12);

    const scalar_t* vec_in = input + vecIdx * C;
    scalar_t* vec_out = output + vecIdx * C;

    // Coalesced write by having each thread write consecutive elements
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        vec_out[i] = vec_in[i] * invNorm;
    }
}

// The forward function launches the two kernel stages

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D for optimized coalesced kernel");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous for optimized coalesced kernel");

    const int total_vectors = input.size(0);
    const int C = input.size(1);

    auto output = torch::empty_like(input);
    auto partial_sums = torch::empty({total_vectors}, input.options());

    // Launch one block per vector
    int threads = 256;
    dim3 grid(total_vectors);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_coalesced_stage1", ([&] {
        l2norm_coalesced_stage1<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_coalesced_stage2", ([&] {
        l2norm_coalesced_stage2<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1 with coalesced memory accesses");
}
