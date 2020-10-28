#include <torch/types.h>
using namespace torch;

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TensorAcc3R PackedTensorAccessor32<scalar_t,3,RestrictPtrTraits>
#define TensorAcc4R PackedTensorAccessor32<scalar_t,4,RestrictPtrTraits>
#define TensorAcc5R PackedTensorAccessor32<scalar_t,5,RestrictPtrTraits>
#define WITHIN_BOUNDS(y, x, H, W) ((y) >= 0 && (y) < (H) && (x) >= 0 && (x) < (W))

#define FORWARD_GROUP_THREADS 16
#define FORWARD_KERNEL_THREADS 8
#define BACKWARD_IN_KERNEL_THREADS 16
#define BACKWARD_W_BATCH_THREADS 4
#define BACKWARD_W_SPATIAL_THREADS 16

#define MAX_KERNEL_SIZE 15

namespace {
template <typename scalar_t>
__global__ void weighted_corr_forward_kernel(
        const TensorAcc4R in1, const TensorAcc4R in2,
        const TensorAcc3R weights, TensorAcc4R output,
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups,
        int batch) {
    /*
     Only support filter size smaller than MAX_KERNEL_SIZE.
    */
    const int spatial_size = in1.size(3);
    const int group_size = in_channel / num_groups;
    const int rad_filter = filter_size / 2;

    // indicies of the current thread
    const int b = batch;
    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int g = blockIdx.z;

    const int kernel_i = threadIdx.x;
    const int kernel_j = threadIdx.y;
    const int thread = threadIdx.z;

    const int group_start = g * group_size;
    const int output_dim_yz = filter_size * num_groups;

    __shared__ scalar_t prod_sum[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][FORWARD_GROUP_THREADS];

    for (int i = kernel_i; i < filter_size; i += FORWARD_KERNEL_THREADS) {
        int y2 = y + dilation * (i - rad_filter);
        for (int j = kernel_j; j < filter_size; j += FORWARD_KERNEL_THREADS) {
            int x2 = x + dilation * (j - rad_filter);
            prod_sum[i][j][thread] = 0;

            if (WITHIN_BOUNDS(y2, x2, spatial_size, spatial_size)) {
                for (int k = thread; k < group_size; k += FORWARD_GROUP_THREADS) {
                    int channel_idx = group_start + k;
                    prod_sum[i][j][thread] += in1[b][channel_idx][y][x] * 
                                              in2[b][channel_idx][y2][x2] * 
                                              weights[channel_idx][i][j];
                }
            }
        }
    }

    __syncthreads();

    if (thread == 0) {
        // aggregate results in the last dimention of prod_sum
        for (int i = kernel_i; i < filter_size; i += FORWARD_KERNEL_THREADS) {
            int output_idx_offset = i * output_dim_yz + g;
            for (int j = kernel_j; j < filter_size; j += FORWARD_KERNEL_THREADS) {
                scalar_t reduce_sum = 0;
                for (int k = 0; k < FORWARD_GROUP_THREADS; ++k) {
                    reduce_sum += prod_sum[i][j][k];
                }
                output[b][output_idx_offset + j * num_groups][y][x] = 
                        reduce_sum / static_cast<scalar_t>(group_size);
            }
        }
    }

}


template <typename scalar_t>
__global__ void weighted_corr_backward_in12_kernel(
        const TensorAcc4R in1, const TensorAcc4R in2, 
        const TensorAcc3R weights, const TensorAcc4R grad_output, 
        TensorAcc4R grad_in1, TensorAcc4R grad_in2, 
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups,
        int batch) {

    const int spatial_size = in1.size(3);
    const int group_size = in_channel / num_groups;
    const int rad_filter = filter_size / 2;

    // indicies of the current thread
    const int b = batch;
    const int c = blockIdx.x;
    const int h = blockIdx.y;
    const int w = blockIdx.z;

    const int kernel_i = threadIdx.x;
    const int kernel_j = threadIdx.y;

    const int output_dim_yz = filter_size * num_groups;
    const int output_z_offset = c / group_size;

    __shared__ scalar_t prod_sum[2][BACKWARD_IN_KERNEL_THREADS * BACKWARD_IN_KERNEL_THREADS];

    const int thread_idx = BACKWARD_IN_KERNEL_THREADS * kernel_i + kernel_j;

    prod_sum[0][thread_idx] = 0;
    prod_sum[1][thread_idx] = 0;

    for (int i = kernel_i; i < filter_size; i += BACKWARD_IN_KERNEL_THREADS) {
        int pos_h_1 = h + dilation * (i - rad_filter);
        int pos_h_2 = h + dilation * (rad_filter - i);

        int output_idx_offset = i * output_dim_yz + output_z_offset;

        for (int j = kernel_j; j < filter_size; j += BACKWARD_IN_KERNEL_THREADS) {
            int pos_w_1 = w + dilation * (j - rad_filter);
            int pos_w_2 = w + dilation * (rad_filter - j);

            int output_idx = output_idx_offset + j * num_groups;

            // accumulate grad_1
            if (WITHIN_BOUNDS(pos_h_1, pos_w_1, spatial_size, spatial_size)) {
                prod_sum[0][thread_idx] += in2[b][c][pos_h_1][pos_w_1] * 
                                                   weights[c][i][j] * 
                                                   grad_output[b][output_idx][h][w];
            }

            // accumulate grad_2
            if (WITHIN_BOUNDS(pos_h_2, pos_w_2, spatial_size, spatial_size)) {
                prod_sum[1][thread_idx] += in1[b][c][pos_h_2][pos_w_2] * 
                                                   weights[c][i][j] * 
                                                   grad_output[b][output_idx][pos_h_2][pos_w_2];
            }
        }
    }

    __syncthreads();

    // sum reduction
    for (int stride = BACKWARD_IN_KERNEL_THREADS * BACKWARD_IN_KERNEL_THREADS / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            prod_sum[0][thread_idx] += prod_sum[0][thread_idx + stride];
            prod_sum[1][thread_idx] += prod_sum[1][thread_idx + stride];
        }
        __syncthreads();
    }
    if (thread_idx == 0) {
        grad_in1[b][c][h][w] = prod_sum[0][0] / static_cast<scalar_t>(group_size);
        grad_in2[b][c][h][w] = prod_sum[1][0] / static_cast<scalar_t>(group_size);
    }

}


template <typename scalar_t>
__global__ void weighted_corr_backward_weights_kernel(
        const TensorAcc4R in1, const TensorAcc4R in2, 
        const TensorAcc3R weights, const TensorAcc4R grad_output, 
        TensorAcc3R grad_weights, 
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups) {

    const int batch_size = in1.size(0);
    const int spatial_size = in1.size(3);
    const int group_size = in_channel / num_groups;
    const int rad_filter = filter_size / 2;

    // indicies of the current thread
    const int c = blockIdx.x;
    const int kernel_i = blockIdx.y;
    const int kernel_j = blockIdx.z;

    const int b_offset = threadIdx.x;
    const int h_offset = threadIdx.y;
    const int w_offset = threadIdx.z;

    const int output_idx = kernel_i * filter_size * num_groups + kernel_j * num_groups + c / group_size;
    const int h_pos_offset = dilation * (kernel_i - rad_filter);
    const int w_pos_offset = dilation * (kernel_j - rad_filter);

    __shared__ scalar_t prod_sum[BACKWARD_W_BATCH_THREADS * BACKWARD_W_SPATIAL_THREADS * BACKWARD_W_SPATIAL_THREADS];

    const int thread_idx = b_offset * BACKWARD_W_SPATIAL_THREADS * BACKWARD_W_SPATIAL_THREADS +
                           h_offset * BACKWARD_W_SPATIAL_THREADS + w_offset;

    prod_sum[thread_idx] = 0;

    for (int b = b_offset; b < batch_size; b += BACKWARD_W_BATCH_THREADS) {
        for (int h = h_offset; h < spatial_size; h += BACKWARD_W_SPATIAL_THREADS) {
            int pos_h = h + h_pos_offset;

            for (int w = w_offset; w < spatial_size; w += BACKWARD_W_SPATIAL_THREADS) {
                int pos_w = w + w_pos_offset;

                if (WITHIN_BOUNDS(pos_h, pos_w, spatial_size, spatial_size)) {
                    prod_sum[thread_idx] += in1[b][c][h][w] * 
                                            in2[b][c][pos_h][pos_w] * 
                                            grad_output[b][output_idx][h][w];
                }
            }
        }

    }

    __syncthreads();
    
    // sum reduction
    for (int stride = BACKWARD_W_BATCH_THREADS * BACKWARD_W_SPATIAL_THREADS * BACKWARD_W_SPATIAL_THREADS / 2;
         stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            prod_sum[thread_idx] += prod_sum[thread_idx + stride];
        }
        __syncthreads();
    }
    if (thread_idx == 0) {
        grad_weights[c][kernel_i][kernel_j] = prod_sum[0] / static_cast<scalar_t>(group_size);
    }
}


}



Tensor weighted_corr_cuda_forward(
        Tensor input1, Tensor input2, Tensor weights,
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups) {

    const int batch_size = input1.size(0);
    const int spatial_size = input1.size(3);

    auto output = torch::zeros(
        {batch_size, num_groups * filter_size * filter_size, spatial_size, spatial_size},
        input1.options()
    );

    auto cont_input1 = input1.contiguous();
    auto cont_input2 = input2.contiguous();
    auto cont_weights = weights.contiguous();

    const dim3 blocks(spatial_size, spatial_size, num_groups);
    const dim3 threads(FORWARD_KERNEL_THREADS, FORWARD_KERNEL_THREADS, FORWARD_GROUP_THREADS);

    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "weighted_corr_forward_cuda", ([&] {
        TensorAcc4R cont_input1_acc = cont_input1.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
        TensorAcc4R cont_input2_acc = cont_input2.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
        TensorAcc3R cont_weights_acc = cont_weights.packed_accessor32<scalar_t,3,RestrictPtrTraits>();
        TensorAcc4R output_acc = output.packed_accessor32<scalar_t,4,RestrictPtrTraits>();

        for (int b = 0; b < batch_size; ++b) {
            weighted_corr_forward_kernel<scalar_t><<<blocks, threads>>>(
                cont_input1_acc, cont_input2_acc, cont_weights_acc, output_acc, 
                in_channel, filter_size, dilation, num_groups, b
            );
        }
    }));

    return output;
}




std::vector<Tensor> weighted_corr_cuda_backward(
        Tensor input1, Tensor input2, Tensor weights, Tensor grad_output, 
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups) {

    const int batch_size = input1.size(0);
    const int spatial_size = input1.size(3);

    auto grad_input1 = torch::zeros_like(input1);
    auto grad_input2 = torch::zeros_like(input2);
    auto grad_weights = torch::zeros_like(weights);

    auto cont_input1 = input1.contiguous();
    auto cont_input2 = input2.contiguous();
    auto cont_weights = weights.contiguous();

    const dim3 blocks_in12(in_channel, spatial_size, spatial_size);
    const dim3 threads_in12(BACKWARD_IN_KERNEL_THREADS, BACKWARD_IN_KERNEL_THREADS);
    const dim3 blocks_w(in_channel, filter_size, filter_size);
    const dim3 threads_w(BACKWARD_W_BATCH_THREADS, BACKWARD_W_SPATIAL_THREADS, BACKWARD_W_SPATIAL_THREADS);

    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "weighted_corr_backward_cuda", ([&] {
        TensorAcc4R cont_input1_acc = cont_input1.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
        TensorAcc4R cont_input2_acc = cont_input2.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
        TensorAcc3R cont_weights_acc = cont_weights.packed_accessor32<scalar_t,3,RestrictPtrTraits>();
        TensorAcc4R grad_input1_acc = grad_input1.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
        TensorAcc4R grad_input2_acc = grad_input2.packed_accessor32<scalar_t,4,RestrictPtrTraits>();
        TensorAcc3R grad_weights_acc = grad_weights.packed_accessor32<scalar_t,3,RestrictPtrTraits>();
        TensorAcc4R grad_output_acc = grad_output.packed_accessor32<scalar_t,4,RestrictPtrTraits>();

        for (int b = 0; b < batch_size; ++b) {
            weighted_corr_backward_in12_kernel<scalar_t><<<blocks_in12, threads_in12>>>(
                cont_input1_acc, cont_input2_acc, cont_weights_acc, grad_output_acc, 
                grad_input1_acc, grad_input2_acc, 
                in_channel, filter_size, dilation, num_groups, b
            );
        }

        weighted_corr_backward_weights_kernel<scalar_t><<<blocks_w, threads_w>>>(
            cont_input1_acc, cont_input2_acc, cont_weights_acc, grad_output_acc, 
            grad_weights_acc, 
            in_channel, filter_size, dilation, num_groups
        );

    }));

    return {grad_input1, grad_input2, grad_weights};
}
