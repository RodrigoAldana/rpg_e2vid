import lpips
import time
from model.model import *
from model.submodules import *
from dataset import EventData, EventSequences
from options.inference_options import set_inference_options
from utils.inference_utils import IntensityRescaler
import argparse

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
            'num_encoders': 3,
            'num_residual_blocks': 2,
            'use_upsample_conv': False,
            'norm': 'BN'}
model = E2VIDRecurrent(net_conf)

# Training configuration
# Important: unroll_L defines how much data must be driven to the GPU at the same time.
# With current GPU, L=40 is not possible. Check nvidia-smi in a terminal to monitor memory usage
training_conf = {'learning_rate': 1e-4,
                 'epochs': 17,
                 'unroll_L': 15,
                 'height': 180,
                 'width': 240,
                 'batch_size': 2,
                 'checkpoint_interval': 100}

# Set GPU as main torch device (if available) and move model to it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function (LPIPS VGG)
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_vgg = loss_fn_vgg.to(device)

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

            # Run the network over the current sequence and evaluate it over L intervals
            for time_interval in range(training_conf['unroll_L']):
                events, reference_frame = dataset.get_item(time_interval)
                events = events.to(device)
                reference_frame = reference_frame.to(device)
                predicted_frame, states = model.forward(events, states)

                # Rescale image to range of 0-255.
                # Same as intensity_rescaler(predicted_frame) but keeping floating point format
                predicted_frame = intensity_rescaler.rescale(predicted_frame)
                loss = loss + loss_fn_vgg(predicted_frame, reference_frame)
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
