__constant__ float c_weight[16384];

__device__ inline float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_X, int BLOCK_Y, bool USE_CONSTANT_MEM>
__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kH, const int kW, const int sH, const int sW,
    const int pH, const int pW
) {
    __shared__ float partial_sums[32];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ow = blockIdx.x * BLOCK_X + tx;
    const int oh = blockIdx.y * BLOCK_Y + ty;
    const int oc = blockIdx.z % C_out;
    const int n = blockIdx.z / C_out;
    
    if (ow >= W_out || oh >= H_out) return;
    
    float sum = 0.0f;
    const int lane = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    #pragma unroll
    for (int ic = 0; ic < C_in; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kH; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kW; ++kw) {
                const int i_val = oh + pH - kh;
                const int j_val = ow + pW - kw;

                if ((i_val % sH == 0) && (j_val % sW == 0)) {
                    const int i_in = i_val / sH;
                    const int j_in = j_val / sW;

                    if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                        const int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                        const int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                        const float w_val = USE_CONSTANT_MEM ? c_weight[weight_idx] : weight[weight_idx];
                        sum += input[input_idx] * w_val;
                    }
                }
            }
        }
    }

    sum = warpReduceSum(sum);
    
    if (lane == 0) {
        partial_sums[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (lane < (blockDim.x * blockDim.y + 31)/32) ? partial_sums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        
        if (lane == 0) {
            if (bias != nullptr) {
                sum += bias[oc];
            }
            const int output_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
            output[output_idx] = sum;
        }
    }
}