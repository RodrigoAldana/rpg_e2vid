import lpips
import time
import math
from model.model import *
from model.submodules import *
from dataset import EventData, EventSequences, flow_warp
from options.inference_options import set_inference_options
from utils.inference_utils import IntensityRescaler
import argparse
import numpy as np
import cv2
import torch

args = argparse.ArgumentParser(description='Training a trained network')
args.add_argument('-t', '--path-to-training-data', required=True, type=str,
                        help='path to training dataset')
args.add_argument('-c', '--path-to-checkpoint', required=True, type=str,
                        help='path to checkpoint folders')
args.add_argument('-m', '--path_to_model', required=True, type=str,help='path to model weights')

set_inference_options(args)
options = args.parse_args()

model = torch.load(options.path_to_model)

# Set GPU as main torch device (if available) and move model to it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Resconstruction loss (LPIPS VGG)
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_vgg = loss_fn_vgg.to(device)

# Warping and parameters for temporal consistency loss
unroll_L = 5
alpha = 0.01
lambda_tc = 0.000 #0.0001
L_0 = 2
m = 15 # Margin to crop warped frames to avoid distorted borders


# Intensity rescaler is needed to be able to match the model output scale to RBG image values
intensity_rescaler = IntensityRescaler(options)

# Instantiate event dataset (only paths to data)
sequences = EventSequences(options.path_to_training_data)
# Training
with torch.no_grad():
    loss = 0
    for seq in sequences:
        print(seq)
        dataset = EventData(root=options.path_to_training_data, seq=seq, width=240, height=180,
                            num_encoders=model.num_encoders, options=options)
        states = None
        current_L = 0
        e, prev_frame, f = dataset.get_item(0, num_encoders=model.num_encoders)
        # Run the network over the current sequence and evaluate it over L intervals
        for time_interval in range(unroll_L):
            events, reference_frame, flow = dataset.get_item(time_interval, num_encoders=model.num_encoders)
            events = events.to(device)
            reference_frame = reference_frame.to(device)
            flow = flow.to(device)

            # Rescale frame to range 0-1
            reference_frame = reference_frame/255

            # Foward pass
            initial_guess = prev_frame.to(device)
            # model.unetrecurrent0.final_activation = True
            # model.unetrecurrent1.final_activation = True
            # predicted_frame, states = model(events, states, 0)
            # predicted_frame, states = model(events, states)
            predicted_frame, states = model(events, states, 6)
            # prev_frame = predicted_frame
            # Rescale image to range of 0-255.
            # Same as intensity_rescaler(predicted_frame) but keeping floating point format
            # predicted_frame = intensity_rescaler.rescale(predicted_frame)

            # Temporal consistency and reconstruction loss
            if (current_L < L_0):
                # Skip temporal consistency loss for the first L0 elements in order for reconstruction to converge
                temporal_consistency_loss = 0
                current_L += 1
            else:
                warpedimg = flow_warp(prev_reference_frame, flow)
                B, C, H, W = warpedimg.size()
                coef = math.exp(-alpha * torch.norm((reference_frame[:, :, m:H - m, m:W - m] - warpedimg[:, :, m:H - m, m:W - m]), 2) ** 2)
                warpedrec = flow_warp(prev_predicted_frame, flow)
                temporal_consistency_loss = coef * torch.norm(
                    (predicted_frame[:, :, m:H - m, m:W - m] - warpedrec[:, :, m:H - m, m:W - m]), 1)
                #+ lambda_tc * temporal_consistency_loss)
            loss = loss + loss_fn_vgg(predicted_frame, reference_frame)


            prev_predicted_frame = predicted_frame
            prev_reference_frame = reference_frame

    print(loss)

