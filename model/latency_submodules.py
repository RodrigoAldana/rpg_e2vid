import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from ttictoc import tic,toc

class dyn_raw_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 input_channels_keep = False, output_channels_keep = False):
        super(dyn_raw_conv, self).__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.stride = stride
        self.padding = padding
        self.weight = conv2d.weight
        self.use_bias = bias
        if bias:
            self.bias = conv2d.bias
        self.latency_mode = 0

        self.kernel = [2,2]
        if input_channels_keep:
            self.kernel[0] = 1
        if output_channels_keep:
            self.kernel[1] = 1

    def set_latency_mode(self, latency_mode):
        self.latency_mode = latency_mode

    def forward(self,x):

        if self.latency_mode == 0:
            w = self.weight
            if self.use_bias:
                b = self.bias
        else:
            w = self.weight.permute((3,2,1,0))
            w = torch.nn.functional.avg_pool2d(w, kernel_size=self.kernel, stride=self.kernel)
            w = w.permute((3,2,1,0))
            if self.use_bias:
                b = torch.unsqueeze(torch.unsqueeze(self.bias,0),0)
                b = torch.nn.functional.avg_pool1d(b, kernel_size=self.kernel[1], stride=self.kernel[1])
                b = torch.squeeze(torch.squeeze(b, 0), 0)
        if self.use_bias:
            x = f.conv2d(x,w,b,stride=self.stride,padding=self.padding)
        else:
            x = f.conv2d(x,w, stride=self.stride, padding=self.padding)
        return x

class dyn_raw_transposed_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,output_padding=1,bias=True,
                 input_channels_keep = False, output_channels_keep = False):
        super(dyn_raw_transposed_conv, self).__init__()
        transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)
        self.stride = stride
        self.padding = padding
        self.weight = transposed_conv2d.weight
        self.use_bias = bias
        self.output_padding = output_padding
        if bias:
            self.bias = transposed_conv2d.bias
        self.latency_mode = 0

        self.kernel = [2,2]
        if input_channels_keep:
            self.kernel[0] = 1
        if output_channels_keep:
            self.kernel[1] = 1

    def set_latency_mode(self, latency_mode):
        self.latency_mode = latency_mode

    def forward(self,x):

        if self.latency_mode == 0:
            w = self.weight
            if self.use_bias:
                b = self.bias
        else:
            w = self.weight.permute((3,2,1,0))
            w = torch.nn.functional.avg_pool2d(w, kernel_size=self.kernel, stride=self.kernel)
            w = w.permute((3,2,1,0))
            if self.use_bias:
                b = torch.unsqueeze(torch.unsqueeze(self.bias,0),0)
                b = torch.nn.functional.avg_pool1d(b, kernel_size=self.kernel[1], stride=self.kernel[1])
                b = torch.squeeze(torch.squeeze(b, 0), 0)

        if self.use_bias:
            x = f.conv_transpose2d(
                x,weight=w,bias=b,stride=self.stride,
                padding=self.padding,output_padding=self.output_padding)
        else:
            x = f.conv_transpose2d(
                x,weight=w,stride=self.stride,
                padding=self.padding,output_padding=self.output_padding)
        return x

class dyn_raw_batchNorm(nn.Module):
    def __init__(self, out_channels):
        super(dyn_raw_batchNorm, self).__init__()
        self.org_out_channels = out_channels
        self.curr_out_channels = out_channels
        norm_layer = nn.BatchNorm2d(out_channels)
        self.bias = norm_layer.bias
        self.register_buffer('running_mean', norm_layer.running_mean)
        self.register_buffer('running_var', norm_layer.running_var)
        # self.running_mean = norm_layer.running_mean
        # self.running_var = norm_layer.running_var
        self.weight = norm_layer.weight
        self.latency_mode = 0

    def set_latency_mode(self, latency_mode):
        self.latency_mode = latency_mode
        if latency_mode == 0:
            self.curr_out_channels = self.org_out_channels
        else:
            self.curr_out_channels = int(self.org_out_channels/2)
        self.running_mean[0:self.curr_out_channels] = torch.zeros(self.curr_out_channels)
        self.running_var[0:self.curr_out_channels] = torch.ones(self.curr_out_channels)

    def forward(self,x):

        if self.latency_mode == 0:
            w = self.weight
            b = self.bias
        else:
            w = torch.unsqueeze(torch.unsqueeze(self.weight,0),0)
            w = torch.nn.functional.avg_pool1d(w, kernel_size=2, stride=2)
            w = torch.squeeze(torch.squeeze(w,0),0)
            b = torch.unsqueeze(torch.unsqueeze(self.bias,0),0)
            b = torch.nn.functional.avg_pool1d(b, kernel_size=2, stride=2)
            b = torch.squeeze(torch.squeeze(b,0),0)


        x = f.batch_norm(x,self.running_mean[0:self.curr_out_channels], self.running_var[0:self.curr_out_channels], w, b)
        return x


class dyn_ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None,
                 input_channels_keep = False, output_channels_keep = False):
        super(dyn_ConvLayer, self).__init__()

        self.conv2d = dyn_raw_conv(in_channels, out_channels, kernel_size, stride, padding,
                                   input_channels_keep=input_channels_keep,output_channels_keep=output_channels_keep)
        self.output_channels_keep = output_channels_keep

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = dyn_raw_batchNorm(out_channels)
        self.latency_mode = 0

    def set_latency_mode(self,latency_mode):
        self.latency_mode = latency_mode
        if self.output_channels_keep == False:
            if self.norm == 'BN':
                self.norm_layer.set_latency_mode(latency_mode)
        self.conv2d.set_latency_mode(latency_mode)

    def forward(self, x):
        out = self.conv2d(x)
        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

class dyn_TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(dyn_TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = dyn_raw_transposed_conv(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = dyn_raw_batchNorm(out_channels)
        self.latency_mode = 0

    def set_latency_mode(self,latency_mode):
        self.latency_mode = latency_mode
        if self.norm == 'BN':
            self.norm_layer.set_latency_mode(latency_mode)
        self.transposed_conv2d.set_latency_mode(latency_mode)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out



class dyn_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm=None):
        super(dyn_ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = dyn_raw_conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = dyn_raw_batchNorm(out_channels)
            self.bn2 = dyn_raw_batchNorm(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = dyn_raw_conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.latency_mode = 0

    def set_latency_mode(self,latency_mode):
        self.latency_mode = latency_mode
        if self.norm == 'BN':
            self.bn1.set_latency_mode(latency_mode)
            self.bn2.set_latency_mode(latency_mode)
        self.conv1.set_latency_mode(latency_mode)
        self.conv2.set_latency_mode(latency_mode)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class dyn_ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/latency_model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(dyn_ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = dyn_ConvLayer(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)
        self.latency_mode = 0

    def set_latency_mode(self,latency_mode):
        self.latency_mode = latency_mode
        self.Gates.set_latency_mode(latency_mode)


    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            if self.latency_mode == 1:
                hidden_size = int(self.hidden_size/2)
            else:
                hidden_size = self.hidden_size

            state_size = tuple([batch_size, hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class dyn_RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None):
        super(dyn_RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type

        self.conv = dyn_ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.recurrent_block = dyn_ConvLSTM(input_size=out_channels, hidden_size=out_channels, kernel_size=3)
        self.latency_mode = 0

        padding_size = int((kernel_size-1)/2)

        self.memory_encoder_hidden = dyn_ConvLayer(in_channels=out_channels, out_channels=int(out_channels/2), kernel_size=kernel_size
                                           , stride=1, padding=padding_size, activation=activation, norm=norm)

        self.memory_decoder_hidden = dyn_ConvLayer(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=kernel_size
                                           , stride=1, padding=padding_size, activation=activation, norm=norm)

        self.memory_encoder_cell = dyn_ConvLayer(in_channels=out_channels, out_channels=int(out_channels/2), kernel_size=kernel_size
                                           , stride=1, padding=padding_size, activation=activation, norm=norm)

        self.memory_decoder_cell = dyn_ConvLayer(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=kernel_size
                                           , stride=1, padding=padding_size, activation=activation, norm=norm)

    def set_latency_mode(self,latency_mode):
        self.latency_mode = latency_mode
        self.conv.set_latency_mode(latency_mode)
        self.recurrent_block.set_latency_mode(latency_mode)

    def forward(self, x, prev_state):

        if self.latency_mode == 1:
            if prev_state != None:
                prev_state_hidden = self.memory_encoder_hidden(prev_state[0])
                prev_state_cell = self.memory_encoder_cell(prev_state[1])
                prev_state = prev_state_hidden, prev_state_cell

        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state

        if self.latency_mode == 1:
            state_hidden = self.memory_decoder_hidden(state[0])
            state_cell = self.memory_decoder_cell(state[1])
            state = state_hidden, state_cell

        return x, state