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
set_inference_options(args)
options = args.parse_args()

# Instantiate model as E2VIDRecurrent
net_conf = {'num_bins': 5,
            'recurrent_block_type': 'convlstm',
            'base_num_channels': 8,
            'num_encoders': 2,
            'num_residual_blocks': 2,
            'use_upsample_conv': False,
            'norm': 'BN'}
model = E2VIDRefine(net_conf)

# Training configuration
# Important: unroll_L defines how much data must be driven to the GPU at the same time.
# With current GPU, L=40 is not possible. Check nvidia-smi in a terminal to monitor memory usage
training_conf = {'learning_rate': 1e-4,
                 'epochs': 160,
                 'unroll_L': 15,
                 'height': 180,
                 'width': 240,
                 'batch_size': 2,
                 'checkpoint_interval': 100,
                 'max_iterations': 5,
                 'use_ref_threshold': 0.9}

# Set GPU as main torch device (if available) and move model to it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Resconstruction loss (LPIPS VGG)
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_vgg = loss_fn_vgg.to(device)

# Warping and parameters for temporal consistency loss
alpha = 0.01
lambda_tc = 0.0001
L_0 = 2
m = 15 # Margin to crop warped frames to avoid distorted borders

# Use ADAM optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=training_conf['learning_rate'])

# Intensity rescaler is needed to be able to match the model output scale to RBG image values
intensity_rescaler = IntensityRescaler(options)

# Instantiate event dataset (only paths to data)
sequences = EventSequences(options.path_to_training_data)
# Training



for t in range(training_conf['epochs']):  # TRAIN FOR 160 EPOCHS
    print('EPOCH ', t)

    # Generates batches from dataset (only paths)
    seq_loader = torch.utils.data.DataLoader(sequences, batch_size=training_conf['batch_size'], shuffle=False)

    for i, batch in enumerate(seq_loader, 0):
        loss = 0

        # Read mini batch data: all file pathnames inside sequences
        for seq in batch:
            dataset = EventData(root=options.path_to_training_data, seq=seq, width=240, height=180,
                                num_encoders=model.num_encoders, options=options)
            states = None
            current_L = 0
            e, prev_frame, f = dataset.get_item(0, num_encoders=model.num_encoders)
            # Run the network over the current sequence and evaluate it over L intervals
            for time_interval in range(training_conf['unroll_L']):
                events, reference_frame, flow = dataset.get_item(time_interval, num_encoders=model.num_encoders)
                events = events.to(device)
                reference_frame = reference_frame.to(device)
                flow = flow.to(device)

                # Rescale frame to range 0-1
                reference_frame = reference_frame/255

                # Foward pass
                iterations = np.random.randint(1,training_conf['max_iterations']+1)
                if np.random.rand() > training_conf['use_ref_threshold']:
                    initial_guess = reference_frame.to(device)
                else:
                    initial_guess = prev_frame.to(device)
                predicted_frame, states = model(events, states, initial_guess, iterations)
                prev_frame = predicted_frame
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

                loss = loss + (loss_fn_vgg(predicted_frame, reference_frame) + lambda_tc * temporal_consistency_loss)

                with torch.no_grad():
                    prev_predicted_frame = predicted_frame
                    prev_reference_frame = reference_frame

            print(seq)

        # Backpropagation
        optimizer.zero_grad()
        loss.sum().backward(retain_graph=True)
        optimizer.step()

        # Checkpoints
        with torch.no_grad():
            if (i+1) % training_conf['checkpoint_interval'] == 0:
                print('Saving')
                time_now = time.time()

                cp_file_name = options.path_to_checkpoint+'cp_' + str(t) + "_" + str(i) + '.tar'
                model_name = options.path_to_checkpoint+'m_' + str(t) + "_" + str(i) + '.pth'

                torch.save({
                    'last_epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, cp_file_name)
                torch.save(model, model_name)

