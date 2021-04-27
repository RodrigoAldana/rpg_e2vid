import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from ttictoc import tic,toc

from .latency_submodules import dyn_ConvLayer, dyn_TransposedConvLayer, dyn_ResidualBlock, dyn_RecurrentConvLayer

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

def skip_sum(x1, x2):
    return x1 + x2

class dyn_BaseUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=False):
        super(dyn_BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(dyn_ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(dyn_TransposedConvLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = dyn_ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm, output_channels_keep = True)


class dyn_UNetRecurrent(dyn_BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(dyn_UNetRecurrent, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = dyn_ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, input_channels_keep = True)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(dyn_RecurrentConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type=recurrent_block_type,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

        self.latency_mode = 0

    def set_latency_mode(self, latency_mode):
        self.latency_mode = latency_mode

        self.head.set_latency_mode(latency_mode)

        for encoder in self.encoders:
            encoder.set_latency_mode(latency_mode)

        for resblock in self.resblocks:
            resblock.set_latency_mode(latency_mode)

        for decoder in self.decoders:
            decoder.set_latency_mode(latency_mode)

        self.pred.set_latency_mode(latency_mode)


    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        # head
        # print("ABOUT TO EVALUATE HEAD")
        # tic()
        x = self.head(x)
        # print("HEAD: ", toc())
        head = x

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        # tic()
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i])
            blocks.append(x)
            states.append(state)
        # print("ENCODERS:", toc())
        # residual blocks
        # tic()
        for resblock in self.resblocks:
            x = resblock(x)
        # print("RESBLOCKS: ", toc())
        # decoder
        # tic()
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))
        # print("DECODER:", toc())

        # tail
        # tic()
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))
        # print("PREDICTION: ", toc())

        return img, states