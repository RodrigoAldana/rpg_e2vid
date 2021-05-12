import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


class BaseUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

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
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)


class UNet(BaseUNet):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNet, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                   num_encoders, base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size, output_size, kernel_size=5,
                                           stride=2, padding=2, norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNetRecurrent, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type=recurrent_block_type,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i])
            blocks.append(x)
            states.append(state)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img, states



class UNetRefine(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNetRefine, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.guess_head = ConvLayer(1, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.build_encoders()
        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

        self.build_guess_encoders()
        self.build_guess_resblocks()
        self.build_guess_decoders()

    def build_encoders(self):
        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    norm=self.norm))
    def build_guess_encoders(self):
        self.guess_encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.guess_encoders.append(RecurrentConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type='convlstm',
                                                    norm=self.norm))
    def build_guess_resblocks(self):
        self.guess_resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.guess_resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_guess_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))
        self.guess_decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.guess_decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))


    def forward(self, x, prev_states, initial_guess):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        # head
        y = self.guess_head(initial_guess)
        x = self.head(x)
        head = y

        y = x+y

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks_x = []
        blocks_y = []
        states = []
        for i in range(self.num_encoders):
            y, state = self.guess_encoders[i](y, prev_states[i])
            x = self.encoders[i](x)
            y = x+y
            blocks_x.append(x)
            blocks_y.append(x)
            states.append(state)

        # residual blocks
        for i in range(self.num_residual_blocks):
            x = self.resblocks[i](x)
            y = self.guess_resblocks[i](y)+x

        # decoder
        for i in range(self.num_encoders):
            x = self.decoders[i](self.apply_skip_connection(x, blocks_x[self.num_encoders - i - 1]))
            y = self.decoders[i](self.apply_skip_connection(y, blocks_y[self.num_encoders - i - 1]))+x
        # tail
        img = self.activation(self.pred(self.apply_skip_connection(y, head)))

        return img, states


class UNetMemFirst(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True,
                 head_layers=4, head_channels=8):

        super(UNetMemFirst, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = nn.ModuleList()
        self.head.append(RecurrentConvLayer(self.num_input_channels, head_channels,
                              kernel_size=5, stride=1, padding=2) )
        for i in range(0, head_layers):
            self.head.append(RecurrentConvLayer(head_channels, head_channels,
                              kernel_size=5, stride=1, padding=2))
        self.head.append(RecurrentConvLayer(head_channels, self.base_num_channels,
                                       kernel_size=5, stride=1, padding=2))
        self.head_layers = head_layers+2
        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """
        if prev_states is None:
            prev_states = [None] * self.head_layers

        # head
        states = []
        for i,head in enumerate(self.head):
            x,state = head(x,prev_states[i])
            states.append(state)
        head = x

        # encoder
        blocks = []
        for encoder in self.encoders:
            x= encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img, states


class UNetHeadless(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True,
                 head_layers=4, head_channels=8):

        super(UNetHeadless, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """
        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for encoder in self.encoders:
            x= encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img