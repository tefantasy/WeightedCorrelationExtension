import torch
import torch.nn as nn
from torch.autograd import Function

import weighted_corr_cuda as corr

class WeightedCorrelationFunc(Function):
    """
    CUDA/C++ extension function wrapper. 
    """
    @staticmethod
    def forward(ctx, input1, input2, weights,
                in_channel, filter_size, dilation=1, num_groups=1):
        ctx.save_for_backward(input1, input2, weights)
        ctx.in_channel = in_channel
        ctx.filter_size = filter_size
        ctx.dilation = dilation
        ctx.num_groups = num_groups

        return corr.forward(input1, input2, weights, 
                            in_channel, filter_size, dilation, num_groups)
    
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, weights = ctx.saved_variables
        in_channel = ctx.in_channel
        filter_size = ctx.filter_size
        dilation = ctx.dilation
        num_groups = ctx.num_groups

        return tuple(corr.backward(input1, input2, weights, grad_output,
                             in_channel, filter_size, dilation, num_groups) + [None] * 4)

class WeightedCorrelation(nn.Module):
    """
    CUDA/C++ extension module wrapper. 
    """
    def __init__(self, in_channel, filter_size, dilation=1, num_groups=1, init_weights=None):
        super(WeightedCorrelation, self).__init__()

        if init_weights is None:
            self.filter_weights = nn.Parameter(torch.Tensor(in_channel, filter_size, filter_size))
            nn.init.kaiming_normal_(self.filter_weights, mode='fan_out', nonlinearity='relu')
        else:
            assert init_weights.shape == (in_channel, filter_size, filter_size), "Wrong weights shape. "
            self.filter_weights = nn.Parameter(init_weights)

        self.in_channel = in_channel
        self.filter_size = filter_size
        self.dilation = dilation
        self.num_groups = num_groups

    def forward(self, input1, input2):
        assert self.filter_weights.is_cuda, "The module is not on GPU. Only support CUDA mode up to now. "

        return WeightedCorrelationFunc.apply(input1, input2, self.filter_weights, 
                        self.in_channel, self.filter_size, self.dilation, self.num_groups)

class WeightedCorrelationLayerExtension(nn.Module):
    """
    Weighted Correlation Layer implemented using CUDA/C++ extension. 
    Currently only support CUDA mode. 
    """

    def __init__(self, in_channel, seq_len, filter_size, dilation=1, num_groups=1):
        """
        Args:
            in_channel: C
            seq_len: L
            filter_size: K. Currently only support K <= 15. 
            dilation: D. If greater than 1, perform dilated correlation.
            num_groups: G. If greater than 1, perform groupwise correlation.
        """
        super(WeightedCorrelationLayerExtension, self).__init__()

        assert dilation >= 1, "Dilation must be greater than 1. "
        assert num_groups >= 1, "Group number must be greater than 1. "
        assert filter_size % 2 == 1, "Only support odd K. "
        assert filter_size <= 15, "Currently only support K <= 15. "
        assert in_channel % num_groups == 0, "Group number must be a divisor of channel number. "

        self.in_channel = in_channel
        self.seq_len = seq_len
        self.dilation = dilation
        self.num_groups = num_groups
        self.span_size = (filter_size - 1) * dilation + 1
        self.pad_size = (self.span_size - 1) // 2

        filter_weights = torch.zeros(in_channel, seq_len, filter_size, filter_size)
        nn.init.kaiming_normal_(filter_weights, mode='fan_out', nonlinearity='relu')

        self.corr_ops = nn.ModuleList()

        for t in range(seq_len):
            corr_op = WeightedCorrelation(
                in_channel, filter_size, dilation=dilation, num_groups=num_groups, 
                init_weights=filter_weights[:, t, :, :]
            )
            self.corr_ops.append(corr_op)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channel, seq, h, w)
        Returns:
            out (Tensor): shape (batch, n_groups*k^2, seq, h, w)
        """
        assert x.is_cuda, "The input tensor is not on GPU. Only support CUDA mode up to now. "
        
        out_list = []
        for t in range(self.seq_len):
            t1, t2 = t - 1, t
            if t1 < 0 : t1 = 0

            out_t = self.corr_ops[t](x[:, :, t1, :, :], x[:, :, t2, :, :])
            out_list.append(out_t.unsqueeze(2))
        return torch.cat(out_list, 2)
