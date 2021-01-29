import torch
import cv2
import numpy
import os
from torch.utils.data import Dataset
from utils.inference_utils import CropParameters, EventPreprocessor


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

    def get_item(self, index):
        event_name = os.path.join(self.event_dir, self.event_tensors[index])  
        frame_name = os.path.join(self.frame_dir, self.frames[index])  
        event_array = numpy.load(event_name)
        event_tensor = torch.tensor(event_array)

        frame = cv2.imread(frame_name)

        frame_tensor = torch.tensor(numpy.transpose(frame, (2, 0, 1)))
        frame_tensor = frame_tensor.type(torch.FloatTensor)

        events = event_tensor.unsqueeze(dim=0)
        events = self.event_preprocessor(events)
        events = self.crop.pad(events)

        pady = torch.zeros((3, 2, 240))
        frame_t = torch.cat((pady, frame_tensor, pady), 1)

        return events, frame_t
