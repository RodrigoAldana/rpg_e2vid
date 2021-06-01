from base import BaseModel
import torch.nn as nn
import torch
from model.unet import UNet, UNetRecurrent, UNetRefine, UNetMemFirst, UNetHeadless
from os.path import join
from model.submodules import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer
from .submodules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU
from ttictoc import tic,toc

class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        assert('num_bins' in config)
        self.num_bins = int(config['num_bins'])  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.head_layers = int(config['head_layers'])
        except KeyError:
            self.head_layers = 3

        try:
            self.head_channels = int(config['head_channels'])
        except KeyError:
            self.head_channels = 8

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True


class E2VID(BaseE2VID):
    def __init__(self, config):
        super(E2VID, self).__init__(config)

        self.unet = UNet(num_input_channels=self.num_bins,
                         num_output_channels=1,
                         skip_type=self.skip_type,
                         activation='sigmoid',
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor), None


class E2VIDRecurrent(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDRecurrent, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states


class E2VIDRefine(BaseE2VID):
    def __init__(self, config):
        super(E2VIDRefine, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins+1,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states, initial_guess, iterations):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """

        for i in range(0,iterations):
            input = torch.cat([event_tensor, initial_guess], dim=1)
            img_pred, states = self.unetrecurrent.forward(input, prev_states)
            initial_guess = img_pred

        return img_pred, states



class E2VIDRefine2(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDRefine2, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unet = UNetRefine(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states, initial_guess, iterations):
        iterations = max(1,iterations)
        for i in range(0,iterations):
            img_pred, states = self.unet(event_tensor, prev_states, initial_guess)
            initial_guess = img_pred

        return img_pred, states


class E2VIDRefine3(BaseE2VID):
    def __init__(self, config):
        super(E2VIDRefine3, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins+1,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states, initial_guess, iterations):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """

        for i in range(0,iterations):
            input = torch.cat([event_tensor, initial_guess], dim=1)
            dimg, states = self.unetrecurrent.forward(input, prev_states)
            img_pred = initial_guess + dimg
            initial_guess = img_pred

        return img_pred, states


class E2VIDRefine4(BaseE2VID):
    def __init__(self, config):
        super(E2VIDRefine4, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetMemFirst(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """

        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)


        return img_pred, states


class E2VIDRefine5(BaseE2VID):
    def __init__(self, config):
        super(E2VIDRefine5, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'


        self.head = nn.ModuleList()
        self.head.append(RecurrentConvLayer(self.num_bins, self.head_channels,
                                            kernel_size=5, stride=1, padding=2))
        for i in range(0, self.head_layers):
            self.head.append(RecurrentConvLayer(self.head_channels, self.head_channels,
                                                kernel_size=5, stride=1, padding=2))
        self.head_layers = self.head_layers + 1

        self.unetrecurrent0 = UNetHeadless(num_input_channels=self.head_channels*self.head_layers,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

        # self.unetrecurrent1 = UNetHeadless(num_input_channels=self.head_channels*self.head_layers,
        #                                    num_output_channels=1,
        #                                    skip_type=self.skip_type,
        #                                    recurrent_block_type=self.recurrent_block_type,
        #                                    activation='sigmoid',
        #                                    num_encoders=self.num_encoders,
        #                                    base_num_channels=int(self.base_num_channels/2),
        #                                    num_residual_blocks=self.num_residual_blocks,
        #                                    norm=self.norm,
        #                                    use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states, mode):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        x = event_tensor
        b,c,w,h = x.size()
        if prev_states is None:
            prev_states = [None] * self.head_layers

        # head
        # tic()
        states = []
        X = torch.zeros((b,0,w,h))
        X = X.to(x.device)
        for i,head in enumerate(self.head):
            x,state = head(x,prev_states[i])
            states.append(state)
            X = torch.cat((X, x), 1)
        # print("HEAD",toc())
        # tic()
        if mode == 0:
            img_pred = self.unetrecurrent0.forward(X)

        if mode == 1:
            img_pred = self.unetrecurrent1.forward(X)
        # print("NET", toc())
        return img_pred, states


class E2VIDRefine6(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDRefine6, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.scale = torch.nn.parameter.Parameter(torch.tensor(0.0))

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

        self.refine = UNetHeadless(num_input_channels=self.num_bins+1,
                                           num_output_channels=+1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='none',
                                           num_encoders=self.num_encoders,
                                           base_num_channels= 16,#int(self.base_num_channels/8),
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states, iterations):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)

        for i in range(0,iterations):
            X = torch.cat((event_tensor,img_pred),1)
            img_pred = img_pred + self.scale*self.refine.forward(X)

        return img_pred, states