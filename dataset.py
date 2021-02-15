import torch
import cv2
import numpy
import os
from torch.utils.data import Dataset
from utils.inference_utils import CropParameters, EventPreprocessor
from torch.nn.functional import grid_sample



class EventSequences(Dataset):

    def __init__(self, root):
        self.sequences = []
        for seq in sorted(os.listdir(root)):
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class EventData:

    def __init__(self, root, seq, width, height, num_encoders, options, transform=None):

        # This members are used to preprocess input events: padding and normalization
        self.crop = CropParameters(width, height, num_encoders)
        self.event_preprocessor = EventPreprocessor(options)

        self.root = root
        self.event_dir = root+seq+'/VoxelGrid-betweenframes-5'
        self.event_tensors = []
        for f in sorted(os.listdir(self.event_dir)):
            if f.endswith('npy'):
                self.event_tensors.append(f)
        if len(self.event_tensors) % 2 != 0:
            self.event_tensors.pop()

        self.frame_dir = root+seq+'/frames'
        self.frames = []
        for f in sorted(os.listdir(self.frame_dir)):
            if f.endswith('png'):
                self.frames.append(f)
        self.transform = transform

        self.flow_dir = root+seq+'/flow'
        self.flow = []
        for f in sorted(os.listdir(self.flow_dir)):
            if f.endswith('npy'):
                self.flow.append(f)

    def get_item(self, index):
        #Take next event/image (index+1) in order to use warp to index -> index+1 to obtain warped image at index+1
        #The flow of index->index+1 is at flow[index]
        event_name = os.path.join(self.event_dir, self.event_tensors[index+1])
        frame_name = os.path.join(self.frame_dir, self.frames[index+1])
        flow_name = os.path.join(self.flow_dir, self.flow[index])

        event_array = numpy.load(event_name)
        event_tensor = torch.tensor(event_array)

        frame = cv2.imread(frame_name)

        flow_array = numpy.load(flow_name)

        frame_tensor = torch.tensor(numpy.transpose(frame, (2, 0, 1)))
        frame_tensor = frame_tensor.type(torch.FloatTensor)

        events = event_tensor.unsqueeze(dim=0)
        events = self.event_preprocessor(events)
        events = self.crop.pad(events)

        pady = torch.zeros((3, 2, 240))
        frame_t = torch.cat((pady, frame_tensor, pady), 1)
        frame_t = frame_t.unsqueeze(dim=0)

        flow_tensor = torch.tensor(numpy.transpose(flow_array, (1,2,0)))
        flow_tensor = flow_tensor.type(torch.FloatTensor)
        padf = torch.zeros((2, 240, 2))
        flow = torch.cat((padf, flow_tensor, padf), 0)
        flow = flow.unsqueeze(dim=0)

        return events, frame_t, flow

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
