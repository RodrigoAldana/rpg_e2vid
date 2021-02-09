import torch
import cv2
import numpy
import os
from torch.utils.data import Dataset
from utils.inference_utils import CropParameters, EventPreprocessor
from scipy.interpolate import interp2d

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
        prev_frame_name = os.path.join(self.frame_dir, self.frames[index])
        flow_name = os.path.join(self.flow_dir, self.flow[index])

        event_array = numpy.load(event_name)
        event_tensor = torch.tensor(event_array)

        frame = cv2.imread(frame_name)

        prev_frame = cv2.imread(prev_frame_name)
        flow_array = numpy.load(flow_name)
        frame_warped = self.warpImage(prev_frame, flow_array[0, :, :], flow_array[1, :, :])

        # Use this piece of code to see how the difference between prev_frame and frame (left) is compared
        # to the warped frame to frame (right). Ideally, warped_frame is close to frame, so the result should be darker
        # fr = numpy.hstack([numpy.abs(prev_frame-frame),numpy.abs(frame_warped-frame)])
        # cv2.imshow('dummy', fr.astype(numpy.uint8))
        # cv2.waitKey(1)

        warped_tensor = torch.tensor(numpy.transpose(frame_warped, (2, 0, 1)))
        warped_tensor = warped_tensor.type(torch.FloatTensor)
        frame_tensor = torch.tensor(numpy.transpose(frame, (2, 0, 1)))
        frame_tensor = frame_tensor.type(torch.FloatTensor)

        events = event_tensor.unsqueeze(dim=0)
        events = self.event_preprocessor(events)
        events = self.crop.pad(events)

        pady = torch.zeros((3, 2, 240))
        frame_t = torch.cat((pady, frame_tensor, pady), 1)
        warped_t = torch.cat((pady, warped_tensor, pady), 1)

        return events, frame_t, warped_t

    #WARP image taken from https://github.com/youngjung/flow-python
    def warpImage(self, im, vx, vy, cast_uint8=True):

        height2, width2, nChannels = im.shape
        height1, width1 = vx.shape

        x = numpy.linspace(1, width2, width2)
        y = numpy.linspace(1, height2, height2)
        X = numpy.linspace(1, width1, width1)
        Y = numpy.linspace(1, height1, height1)
        xx, yy = numpy.meshgrid(x, y)
        XX, YY = numpy.meshgrid(X, Y)
        XX = XX + vx
        YY = YY + vy

        XX = numpy.clip(XX, 1, width2)
        YY = numpy.clip(XX, 1, height2)

        warpI2 = numpy.zeros((height1, width1, nChannels))
        for i in range(nChannels):
            f = interp2d(x, y, im[:, :, i], 'cubic')
            foo = f(X, Y)
            warpI2[:, :, i] = foo

        return warpI2