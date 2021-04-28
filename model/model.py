from base import BaseModel
import torch.nn as nn
import torch
from model.unet import UNet, UNetRecurrent
from os.path import join
from model.submodules import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer


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

class E2VIDSwitching(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDSwitching, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetBase = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

        self.unetReduced = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=int(self.base_num_channels/2),
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

        self.translator = ConvLayer

        self.translator_Base_Reduced_hidden = nn.ModuleList()
        self.translator_Base_Reduced_cell= nn.ModuleList()
        self.translator_Reduced_Base_hidden = nn.ModuleList()
        self.translator_Reduced_Base_cell= nn.ModuleList()


        for input_size, output_size in zip(self.unetBase.encoder_output_sizes, self.unetReduced.encoder_output_sizes):
            self.translator_Base_Reduced_hidden.append(self.translator(input_size, output_size, kernel_size=5,
                                                   stride=1, padding=2))
            self.translator_Base_Reduced_cell.append(self.translator(input_size, output_size, kernel_size=5,
                                                   stride=1, padding=2))

        self.translator_Reduced_Base = nn.ModuleList()
        for input_size, output_size in zip(self.unetReduced.encoder_output_sizes, self.unetBase.encoder_output_sizes ):
            self.translator_Reduced_Base_hidden.append(self.translator(input_size, output_size, kernel_size=5,
                                                   stride=1, padding=2))
            self.translator_Reduced_Base_cell.append(self.translator(input_size, output_size, kernel_size=5,
                                                   stride=1, padding=2))
        self.translation_loss_criterion = torch.nn.MSELoss()



    def forward(self, event_tensor, prev_statesBase, prev_statesReduced):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_predBase, statesBase = self.unetBase.forward(event_tensor, prev_statesBase)
        img_predReduced, statesReduced = self.unetReduced.forward(event_tensor, prev_statesReduced)

        Base_Reduced = self.translate_Base2Reduced(statesBase)
        Reduced_Base = self.translate_Reduced2Based(statesReduced)

        return img_predBase, img_predReduced, statesBase, statesReduced, Base_Reduced, Reduced_Base

    def translation_loss(self, base, reduced, Base_Reduced, Reduced_Base):
        loss = 0
        for i in range(0,self.num_encoders):
            loss = loss + self.translation_loss_criterion(base[i][0], Reduced_Base[i][0])
            loss = loss + self.translation_loss_criterion(base[i][1], Reduced_Base[i][1])
            loss = loss + self.translation_loss_criterion(reduced[i][0], Base_Reduced[i][0])
            loss = loss + self.translation_loss_criterion(reduced[i][1], Base_Reduced[i][1])
        return loss

    def forward_base(self, event_tensor, prev_states):
        img_pred, states = self.unetBase.forward(event_tensor, prev_states)
        return img_pred, states

    def forward_reduced(self, event_tensor, prev_states):
        img_pred, states = self.unetReduced.forward(event_tensor, prev_states)
        return img_pred, states

    def translate_Base2Reduced(self, statesBase):
        if statesBase == None:
            Base_Reduced = None
        else:
            Base_Reduced = []
            for i in range(0,self.num_encoders):
                hidden = self.translator_Base_Reduced_hidden[i](statesBase[i][0])
                cell = self.translator_Base_Reduced_cell[i](statesBase[i][1])
                Base_Reduced.append([hidden, cell])
        return Base_Reduced

    def translate_Reduced2Based(self, statesReduced):
        if statesReduced == None:
            Reduced_Base = None
        else:
            Reduced_Base = []
            for i in range(0,self.num_encoders):
                hidden = self.translator_Reduced_Base_hidden[i](statesReduced[i][0])
                cell = self.translator_Reduced_Base_cell[i](statesReduced[i][1])
                Reduced_Base.append([hidden, cell])
        return Reduced_Base